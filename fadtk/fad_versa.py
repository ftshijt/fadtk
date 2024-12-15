import datetime
import functools
import traceback
from typing import NamedTuple, Union, Dict, Any
import numpy as np
import torch
import torchaudio
from scipy import linalg
from numpy.lib.scimath import sqrt as scisqrt
from pathlib import Path
from hypy_utils import write
from hypy_utils.tqdm_utils import tq, tmap
from hypy_utils.logging_utils import setup_logger

from .kid import compute_kernel_distance as calc_kernel_distance
from .model_loader import ModelLoader
from .utils import *

log = setup_logger()


class FADInfResults(NamedTuple):
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]


def calc_embd_statistics(embd_lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings.
    """
    if len(embd_lst) == 1:
        return embd_lst[0], np.zeros((embd_lst[0].shape[-1], embd_lst[0].shape[-1]))
    return np.mean(embd_lst, axis=0), np.cov(embd_lst, rowvar=False)


def calc_frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- cov1: The covariance matrix over activations for generated samples.
    -- cov2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)

    assert mu1.shape == mu2.shape, \
        f'Training and test mean vectors have different lengths ({mu1.shape} vs {mu2.shape})'
    assert cov1.shape == cov2.shape, \
        f'Training and test covariances have different dimensions ({cov1.shape} vs {cov2.shape})'

    diff = mu1 - mu2

    # Product might be almost singular
    # NOTE: issues with sqrtm for newer scipy versions
    # using eigenvalue method as workaround
    covmean_sqrtm, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    
    # eigenvalue method
    D, V = linalg.eig(cov1.dot(cov2))
    covmean = (V * scisqrt(D)) @ linalg.inv(V)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        log.info(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    tr_covmean_sqrtm = np.trace(covmean_sqrtm)
    if np.iscomplexobj(tr_covmean_sqrtm):
        if np.abs(tr_covmean_sqrtm.imag) < 1e-3:
            tr_covmean_sqrtm = tr_covmean_sqrtm.real

    if not(np.iscomplexobj(tr_covmean_sqrtm)):
        delt = np.abs(tr_covmean - tr_covmean_sqrtm)
        if delt > 1e-3:
            log.warning(f'Detected high error in sqrtm calculation: {delt}')

    return (diff.dot(diff) + np.trace(cov1)
            + np.trace(cov2) - 2 * tr_covmean)


class FrechetAudioDistance:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded = False

    def __init__(self, ml: ModelLoader, audio_load_worker=8, load_model=True):
        self.ml = ml
        self.audio_load_worker = audio_load_worker

        if load_model:
            self.ml.load_model()
            self.loaded = True

        # Disable gradient calculation because we're not training
        torch.autograd.set_grad_enabled(False)

    def load_audio(self, f: Union[str, Path, np.array], cache_dir: PathLike, fsorig: int = 16000, audio_key: str = None):

        # Create a directory for storing normalized audio files
        cache_dir = Path(cache_dir) / "convert" / str(self.ml.sr)
        if audio_key is not None:
            new = (cache_dir / audio_key).with_suffix(".wav")
        else:
            random_filename = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            new = (cache_dir / random_filename).with_suffix(".wav")

        # For case of raw audio, directly return (assume preprocessed by versa)
        if type(f) is tuple:
            x = torch.tensor(f[1]) # return the audio data (from (fs, audio) tuple)
        elif type(f) is not str and type(f) is not Path:
            x = torch.tensor(f)
        else:
            x = None
            f = Path(f)

        if not new.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            if x is None:
                x, fsorig = torchaudio.load(str(f))
            if len(x.shape) == 2:
                x = torch.mean(x.float(),1).unsqueeze(0) # convert to mono
            else:
                x = x.unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(
                fsorig,
                self.ml.sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            y = resampler(x)
            torchaudio.save(new, y, self.ml.sr, encoding="PCM_S", bits_per_sample=16)
        return self.ml.load_wav(new)

    def cache_embedding_file(self, audio_key: str, audio_dir: PathLike, cache_dir: PathLike, fsorig: int = 16000):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        cache = get_cache_embedding_path_versa(self.ml.name, audio_key, cache_dir)

        if cache.exists():
            return

        # Load file, get embedding, save embedding
        wav_data = self.load_audio(audio_dir, cache_dir, fsorig, audio_key)
        embd = self.ml.get_embedding(wav_data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)
    
    def calculate_embedding_online(self, audio_dir: PathLike, cache_dir: PathLike, fsorig: int = 16000):
        """
        Compute embedding for an audio file online, return the embedding
        """
        wav_data = self.load_audio(audio_dir, cache_dir, fsorig, None)
        embd = self.ml.get_embedding(wav_data)
        return embd

    def read_embedding_file(self, audio_key: str, cache_dir: PathLike):
        """
        Read embedding from a cached file.
        """
        cache = get_cache_embedding_path_versa(self.ml.name, audio_key, cache_dir)
        if not str(cache).endswith(".npy"):
            cache = Path(str(cache) + ".npy")
        assert cache.exists(), f"Embedding file {cache} does not exist, please run cache_embedding_file first."
        return np.load(cache)
    
    def load_embeddings(self, audio_files: Dict[str, PathLike], cache_dir: PathLike,  max_count: int = -1, concat: bool = True, return_path: bool = False):
        """
        Load embeddings for all audio files in a directory.
        """
        log.info(f"Loading {len(audio_files.keys())} audio files from {dir}...")

        return self._load_embeddings(audio_files,  cache_dir, max_count=max_count, concat=concat, return_path=return_path)

    def _load_embeddings(self, audio_files: Dict[str, PathLike], cache_dir: Union[str, Path], max_count: int = -1, concat: bool = True, return_path: bool = False):
        """
        Load embeddings for a list of audio files.
        """
        if len(audio_files.keys()) == 0:
            raise ValueError("No files provided")
        if return_path:
            embd_list = [get_cache_embedding_path_versa(self.ml.name, audio_key, cache_dir) for audio_key in audio_files.keys()]
            embd_list = [str(e) + ".npy" if not str(e).endswith(".npy") else e for e in embd_list]
            return embd_list
        # Load embeddings
        if max_count == -1:
            audio_keys = list(audio_files.keys())
            embd_lst = tmap(functools.partial(self.read_embedding_file, cache_dir=cache_dir), audio_keys, desc="Loading audio files...", max_workers=self.audio_load_worker)
        else:
            total_len = 0
            embd_lst = []
            for f in tq(audio_files.keys(), "Loading files"):
                embd_lst.append(self.read_embedding_file(f, cache_dir))
                total_len += embd_lst[-1].shape[0]
                if total_len > max_count:
                    break
        
        # Concatenate embeddings if needed
        if concat:
            return np.concatenate(embd_lst, axis=0)
        else:
            return embd_lst, audio_keys
    
    def load_stats(self, eval_files: Dict[str, Any], cache_dir: Union[Path, str], data_type="baseline"):
        """
        Load embedding statistics from a directory.
        """

        stats_dir = Path(cache_dir) / data_type / "stats" / self.ml.name
        if stats_dir.exists():
            log.info(f"Embedding statistics is already cached for {cache_dir}, loading...")
            mu = np.load(stats_dir / "mu.npy")
            cov = np.load(stats_dir / "cov.npy")
            return mu, cov

        log.info(f"Loading embedding files from {cache_dir}...")
        
        embd_list = self.load_embeddings(eval_files, Path(cache_dir) / data_type, max_count=-1, return_path=True)
        mu, cov = calculate_embd_statistics_online(embd_list)
        log.info("> Embeddings statistics calculated.")

        # Save statistics
        stats_dir.mkdir(parents=True, exist_ok=True)
        np.save(stats_dir / "mu.npy", mu)
        np.save(stats_dir / "cov.npy", cov)
        
        return mu, cov

    def score(self, baseline_files: Dict[str, Any], eval_files: Dict[str, Any], cache_dir: Union[Path, str]):
        """
        Calculate a single FAD score between a background and an eval set.

        :param baseline_files: baseline audio audios
        :param eval_files: eval audio files
        :param cache_dir: cache directory for stats and embeddings
        """
        mu_bg, cov_bg = self.load_stats(baseline_files, cache_dir, data_type="baseline")
        mu_eval, cov_eval = self.load_stats(eval_files, cache_dir, data_type="eval")

        return calc_frechet_distance(mu_bg, cov_bg, mu_eval, cov_eval)

    def score_inf(self, baseline_files: Dict[str, Any], eval_files: Dict[str, Any], cache_dir: Union[Path, str], steps: int = 25, min_n = 500):
        """
        Calculate FAD for different n (number of samples) and compute FAD-inf.

        :param baseline_files: baseline audio audios
        :param eval_files: eval audio files
        :param cache_dir: cache directory for stats and embeddings
        :param steps: number of steps to use
        :param min_n: minimum n to use
        """
        log.info(f"Calculating FAD-inf for {self.ml.name}...")
        # 1. Load background embeddings
        mu_base, cov_base = self.load_stats(baseline_files, cache_dir, data_type="baseline")
        embeds = self._load_embeddings(eval_files, Path(cache_dir) / "eval", concat=True)
        
        # Calculate maximum n
        max_n = len(embeds)
        min_n = min(min_n, max_n)
        steps = min(steps, int(max_n - min_n) + 1)

        # Generate list of ns to use
        ns = [int(n) for n in np.linspace(min_n, max_n, steps)]
        
        results = []
        for n in tq(ns, desc="Calculating FAD-inf"):
            # Select n feature frames randomly (with replacement)
            indices = np.random.choice(embeds.shape[0], size=n, replace=True)
            embds_eval = embeds[indices]
            
            mu_eval, cov_eval = calc_embd_statistics(embds_eval)
            fad_score = calc_frechet_distance(mu_base, cov_base, mu_eval, cov_eval)

            # Add to results
            results.append([n, fad_score])

        # Compute FAD-inf based on linear regression of 1/n
        ys = np.array(results)
        xs = 1 / np.array(ns)
        slope, intercept = np.polyfit(xs, ys[:, 1], 1)

        # Compute R^2
        r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2)

        # Since intercept is the FAD-inf, we can just return it
        return FADInfResults(score=intercept, slope=slope, r2=r2, points=results)
    
    def score_kid(self, baseline_files: Dict[str, Any], eval_files: Dict[str, Any], cache_dir: Union[Path, str], n_subsets: int = 100, subset_size: int = 1000):
        """
        Calculate the KID score for a set of eval files.

        :param baseline_files: baseline audio audios
        :param eval_files: eval audio files
        :param cache_dir: cache directory for stats and embeddings
        :param n_subsets: number of subsets to use
        :param subset_size: size of each subset
        """
        log.info(f"Calculating KID for {self.ml.name}...")

        # 1. Load background embeddings
        baseline_embeds = self._load_embeddings(baseline_files, Path(cache_dir) / "baseline", concat=True)
        embeds = self._load_embeddings(eval_files, Path(cache_dir) / "eval", concat=True)

        # 2. Generate random subsets
        kid_score = calc_kernel_distance(embeds, baseline_embeds, kernel="rbf", kid_subsets=n_subsets, kid_subset_size=subset_size)

        return kid_score
    
    def score_individual(self, baseline_files: Dict[str, Any], eval_audio_path, cache_dir: Union[Path, str], csv_name: Union[Path, str]) -> Path:
        """
        Calculate the FAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline_files: baseline audio audios
        :param eval_files: eval audio files
        :param cache_dir: cache directory for stats and embeddings
        """

        # 1. Load background embeddings
        mu, cov = self.load_stats(baseline_files, cache_dir, data_type="baseline")

        # 2. Define helper function for calculating z score
        def _find_z_helper(f):
            try:
                # Calculate FAD for individual songs
                embd = self.calculate_embedding_online(f)
                mu_eval, cov_eval = calc_embd_statistics(embd)
                return calc_frechet_distance(mu, cov, mu_eval, cov_eval)

            except Exception as e:
                traceback.print_exc()
                log.error(f"An error occurred calculating individual FAD using model {self.ml.name} on file {f}")
                log.error(e)

        # 3. Calculate z score for each eval file
        return _find_z_helper(eval_audio_path)

    
    def score_individual_legacy(self, baseline_files: Dict[str, Any], eval_files: Dict[str, Any], cache_dir: Union[Path, str], csv_name: Union[Path, str]) -> Path:
        """
        Calculate the FAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline_files: baseline audio audios
        :param eval_files: eval audio files
        :param cache_dir: cache directory for stats and embeddings
        :param csv_name: Name of the csv file to write the results to
        :return: Path to the csv file
        """
        csv = Path(csv_name)
        if isinstance(csv_name, str):
            csv = Path('data') / f'fad-individual' / self.ml.name / csv_name
        if csv.exists():
            log.info(f"CSV file {csv} already exists, exiting...")
            return csv

        # 1. Load background embeddings
        mu, cov = self.load_stats(baseline_files, cache_dir, data_type="baseline")

        # 2. Define helper function for calculating z score
        def _find_z_helper(f):
            try:
                # Calculate FAD for individual songs
                embd = self.read_embedding_file(f, cache_dir=cache_dir)
                mu_eval, cov_eval = calc_embd_statistics(embd)
                return calc_frechet_distance(mu, cov, mu_eval, cov_eval)

            except Exception as e:
                traceback.print_exc()
                log.error(f"An error occurred calculating individual FAD using model {self.ml.name} on file {f}")
                log.error(e)

        # 3. Calculate z score for each eval file
        _files = eval_files
        scores = tmap(_find_z_helper, _files, desc=f"Calculating scores", max_workers=self.audio_load_worker)

        # 4. Write the sorted z scores to csv
        pairs = list(zip(_files, scores))
        pairs = [p for p in pairs if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        write(csv, "\n".join([",".join([str(x).replace(',', '_') for x in row]) for row in pairs]))

        return csv
