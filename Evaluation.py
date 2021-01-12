#This code is from https://github.com/craffel/mir_eval
import librosa
from Hparams import HParams
import os
import numpy as np
import scipy
import itertools
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

class Evaluation:
    def __init__(self,h_params:HParams):
        self.h_params = h_params
    
    def evaluate_test_set(self):
        estimated_path = os.path.join(self.h_params.evaluate.input_path,self.h_params.evaluate.model_name)
        reference_path = os.path.join(self.h_params.evaluate.input_path,"gt_vocal_accom")

        estimated_vocal_list = []
        for fname in os.listdir(estimated_path):
            if "vocal.wav" in fname:
                estimated_vocal_list.append(fname)

        total_vocal_sdr = np.array([])
        total_accom_sdr = np.array([])
        total_vocal_sir = np.array([])
        total_accom_sir = np.array([])

        for e_vocal in estimated_vocal_list:
            audio_name = e_vocal[14:].replace("_mix_vocal.wav","")
            estimated_vocal_path = os.path.join(estimated_path,e_vocal)
            estimated_accom_path = os.path.join(estimated_path,e_vocal.replace("_vocal.wav","_accom.wav"))
            reference_vocal_path = os.path.join(reference_path,audio_name + "_vocal.wav")
            reference_accom_path = os.path.join(reference_path,audio_name + "_accom.wav")

            result = self.sdr_sir_sar_popt_eval(ref_vocal_path = reference_vocal_path,
                                                ref_accom_path = reference_accom_path,
                                                es_vocal_path = estimated_vocal_path,
                                                es_accom_path = estimated_accom_path
                                                )
            vocal_sdr = result[0][0]
            accom_sdr = result[0][1]
            vocal_sir = result[1][0]
            accom_sir = result[1][1]
            sir = result[1]
            sar = result[2]
            popt = result[3]
            print(f"{audio_name} Vocal SDR: {vocal_sdr} / Accom SDR: {accom_sdr} / Vocal SIR: {vocal_sir} / Accom SIR {accom_sir}")
            
            total_vocal_sdr = np.append(total_vocal_sdr,vocal_sdr)
            total_accom_sdr = np.append(total_accom_sdr,accom_sdr)
            total_vocal_sir = np.append(total_vocal_sir,vocal_sir)
            total_accom_sir = np.append(total_accom_sir,accom_sir)
            total_accom_sir = np.append(total_accom_sir,sir)
        
        print(f"Mean Vocal SDR : {np.mean(total_vocal_sdr)}")
        print(f"Mean Accom SDR : {np.mean(total_accom_sdr)}")
        print(f"Mean Vocal SIR : {np.mean(total_vocal_sir)}")
        print(f"Mean Accom SIR : {np.mean(total_accom_sir)}")


    def sdr_sir_sar_popt_eval(self,ref_vocal_path,ref_accom_path,es_vocal_path,es_accom_path):
        ref_vocal,_ = librosa.load(ref_vocal_path,sr=self.h_params.preprocess.sample_rate)
        ref_accom,_ = librosa.load(ref_accom_path,sr=self.h_params.preprocess.sample_rate)
        es_vocal,_ = librosa.load(es_vocal_path,sr=self.h_params.preprocess.sample_rate)
        es_accom,_ = librosa.load(es_accom_path,sr=self.h_params.preprocess.sample_rate)
        assert (len(es_vocal)==len(es_accom))
        assert (len(ref_vocal)==len(ref_accom))
        if len(es_vocal) < len(ref_vocal):
            ref_vocal = ref_vocal[:len(es_vocal)]
            ref_accom = ref_accom[:len(es_accom)]
        return self.bss_eval_sources(np.array([ref_vocal,ref_accom]),np.array([es_vocal,es_accom]))

    def bss_eval_sources(self,reference_sources, estimated_sources,
                     compute_permutation=True):
        """
        Ordering and measurement of the separation quality for estimated source
        signals in terms of filtered true source, interference and artifacts.
        The decomposition allows a time-invariant filter distortion of length
        512, as described in Section III.B of [#vincent2006performance]_.
        Passing ``False`` for ``compute_permutation`` will improve the computation
        performance of the evaluation; however, it is not always appropriate and
        is not the way that the BSS_EVAL Matlab toolbox computes bss_eval_sources.
        Examples
        --------
        >>> # reference_sources[n] should be an ndarray of samples of the
        >>> # n'th reference source
        >>> # estimated_sources[n] should be the same for the n'th estimated
        >>> # source
        >>> (sdr, sir, sar,
        ...  perm) = mir_eval.separation.bss_eval_sources(reference_sources,
        ...                                               estimated_sources)
        Parameters
        ----------
        reference_sources : np.ndarray, shape=(nsrc, nsampl)
            matrix containing true sources (must have same shape as
            estimated_sources)
        estimated_sources : np.ndarray, shape=(nsrc, nsampl)
            matrix containing estimated sources (must have same shape as
            reference_sources)
        compute_permutation : bool, optional
            compute permutation of estimate/source combinations (True by default)
        Returns
        -------
        sdr : np.ndarray, shape=(nsrc,)
            vector of Signal to Distortion Ratios (SDR)
        sir : np.ndarray, shape=(nsrc,)
            vector of Source to Interference Ratios (SIR)
        sar : np.ndarray, shape=(nsrc,)
            vector of Sources to Artifacts Ratios (SAR)
        perm : np.ndarray, shape=(nsrc,)
            vector containing the best ordering of estimated sources in
            the mean SIR sense (estimated source number ``perm[j]`` corresponds to
            true source number ``j``). Note: ``perm`` will be ``[0, 1, ...,
            nsrc-1]`` if ``compute_permutation`` is ``False``.
        References
        ----------
        .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
            Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
            Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
            (2007-2010): Achievements and remaining challenges", Signal Processing,
            92, pp. 1928-1936, 2012.
        """

        # make sure the input is of shape (nsrc, nsampl)
        if estimated_sources.ndim == 1:
            estimated_sources = estimated_sources[np.newaxis, :]
        if reference_sources.ndim == 1:
            reference_sources = reference_sources[np.newaxis, :]

        self.validate(reference_sources, estimated_sources)
        # If empty matrices were supplied, return empty lists (special case)
        if reference_sources.size == 0 or estimated_sources.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        nsrc = estimated_sources.shape[0]

        # does user desire permutations?
        if compute_permutation:
            # compute criteria for all possible pair matches
            sdr = np.empty((nsrc, nsrc))
            sir = np.empty((nsrc, nsrc))
            sar = np.empty((nsrc, nsrc))
            for jest in range(nsrc):
                for jtrue in range(nsrc):
                    s_true, e_spat, e_interf, e_artif = \
                        self._bss_decomp_mtifilt(reference_sources,
                                            estimated_sources[jest],
                                            jtrue, 512)
                    sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = \
                        self._bss_source_crit(s_true, e_spat, e_interf, e_artif)

            # select the best ordering
            perms = list(itertools.permutations(list(range(nsrc))))
            mean_sir = np.empty(len(perms))
            dum = np.arange(nsrc)
            for (i, perm) in enumerate(perms):
                mean_sir[i] = np.mean(sir[perm, dum])
            popt = perms[np.argmax(mean_sir)]
            idx = (popt, dum)
            return (sdr[idx], sir[idx], sar[idx], np.asarray(popt))
        else:
            # compute criteria for only the simple correspondence
            # (estimate 1 is estimate corresponding to reference source 1, etc.)
            sdr = np.empty(nsrc)
            sir = np.empty(nsrc)
            sar = np.empty(nsrc)
            for j in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    self._bss_decomp_mtifilt(reference_sources,
                                        estimated_sources[j],
                                        j, 512)
                sdr[j], sir[j], sar[j] = \
                    self._bss_source_crit(s_true, e_spat, e_interf, e_artif)

            # return the default permutation for compatibility
            popt = np.arange(nsrc)
            return (sdr, sir, sar, popt)
    
    def validate(self,reference_sources, estimated_sources):
        """Checks that the input data to a metric are valid, and throws helpful
        errors if not.
        Parameters
        ----------
        reference_sources : np.ndarray, shape=(nsrc, nsampl)
            matrix containing true sources
        estimated_sources : np.ndarray, shape=(nsrc, nsampl)
            matrix containing estimated sources
        """

        if reference_sources.shape != estimated_sources.shape:
            raise ValueError('The shape of estimated sources and the true '
                         'sources should match.  reference_sources.shape '
                         '= {}, estimated_sources.shape '
                         '= {}'.format(reference_sources.shape,
                                       estimated_sources.shape))

        if reference_sources.ndim > 3 or estimated_sources.ndim > 3:
            raise ValueError('The number of dimensions is too high (must be less '
                         'than 3). reference_sources.ndim = {}, '
                         'estimated_sources.ndim '
                         '= {}'.format(reference_sources.ndim,
                                       estimated_sources.ndim))

    def _bss_decomp_mtifilt(self,reference_sources, estimated_source, j, flen):
        """Decomposition of an estimated source image into four components
        representing respectively the true source image, spatial (or filtering)
        distortion, interference and artifacts, derived from the true source
        images using multichannel time-invariant filters.
        """
        nsampl = estimated_source.size
        # decomposition
        # true source image
        s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))
        # spatial (or filtering) distortion
        e_spat = self._project(reference_sources[j, np.newaxis, :], estimated_source,
                          flen) - s_true
        # interference
        e_interf = self._project(reference_sources,
                        estimated_source, flen) - s_true - e_spat
        # artifacts
        e_artif = -s_true - e_spat - e_interf
        e_artif[:nsampl] += estimated_source
        return (s_true, e_spat, e_interf, e_artif)
    
    def _bss_source_crit(self,s_true, e_spat, e_interf, e_artif):
        """Measurement of the separation quality for a given source in terms of
        filtered true source, interference and artifacts.
        """
        # energy ratios
        s_filt = s_true + e_spat
        sdr = self._safe_db(np.sum(s_filt**2), np.sum((e_interf + e_artif)**2))
        sir = self._safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
        sar = self._safe_db(np.sum((s_filt + e_interf)**2), np.sum(e_artif**2))
        return (sdr, sir, sar)

    def _project(self,reference_sources, estimated_source, flen):
        """Least-squares projection of estimated source on the subspace spanned by
        delayed versions of reference sources, with delays between 0 and flen-1
        """
        nsrc = reference_sources.shape[0]
        nsampl = reference_sources.shape[1]

        # computing coefficients of least squares problem via FFT ##
        # zero padding and FFT of input data
        reference_sources = np.hstack((reference_sources,
                                    np.zeros((nsrc, flen - 1))))
        estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
        n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
        sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
        sef = scipy.fftpack.fft(estimated_source, n=n_fft)
        # inner products between delayed versions of reference_sources
        G = np.zeros((nsrc * flen, nsrc * flen))
        for i in range(nsrc):
            for j in range(nsrc):
                ssf = sf[i] * np.conj(sf[j])
                ssf = np.real(scipy.fftpack.ifft(ssf))
                ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                            r=ssf[:flen])
                G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
                G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
        # inner products between estimated_source and delayed versions of
        # reference_sources
        D = np.zeros(nsrc * flen)
        for i in range(nsrc):
            ssef = sf[i] * np.conj(sef)
            ssef = np.real(scipy.fftpack.ifft(ssef))
            D[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

        # Computing projection
        # Distortion filters
        try:
            C = np.linalg.solve(G, D).reshape(flen, nsrc, order='F')
        except np.linalg.linalg.LinAlgError:
            C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order='F')
        # Filtering
        sproj = np.zeros(nsampl + flen - 1)
        for i in range(nsrc):
            sproj += fftconvolve(C[:, i], reference_sources[i])[:nsampl + flen - 1]
        return sproj

    def _safe_db(self,num, den):
        """Properly handle the potential +Inf db SIR, instead of raising a
        RuntimeWarning. Only denominator is checked because the numerator can never
        be 0.
        """
        if den == 0:
            return np.Inf
        return 10 * np.log10(num / den)
