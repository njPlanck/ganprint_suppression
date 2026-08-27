# Using Auto-Encoder To Extracting Or Suppress GAN Model Fingerprints
Our work here is to train an autoencoder to  extract or suppress model fingerprints from synthetic images.
In the Presentation Attack Detection(PAD) system, we were able to confirm the presence of artifacts in GAN generated finger-vein images. We suspect these negatively affected our classifier trained on the fourier features (energy bands), as the real image data were progressively replaced and augumented with synthetic data. In other words, the classifier learned more to the embedded artifacts in the frequency domain of the synthetic data than the spatial structural similarities of these synthetic images to the real ones.

This was problematic because even though the classifier could distinguish real and synthetic samples, it failed to discriminate bonafide and spoofed samples. Earlier we had tried to remove these artifacts using spatial filtering technigues by treating these model fingerprints as Photo Response Non-Uniformity(PRNU) pattern noise. But this has not been very helpful. It is observed these filters only captured the high frequency noise and ignored the low frequency components, such that suppression was minimal and not enough to effectively fool the classifier during detection.


# Encoder Structure 

The design of the encoder structure leverages on the convolutional autoencoder structure for its feature extraction capabilites to denoise GAN generated finger vein images. Like Neves et al hypothesizes, these fingerprints can be removed by this autoencoder, which acts as a non-linear low-pass filter.

During training, the real image samples are fed into the network, and a 3 by 3 kernel is used to extract features before downsampling at every layer of the network on the encoder section. To create the bottleneck, the window size is then reduced and capture the localised featured of the downsampled images in the latent space. These latent features are then reconstructed on the decoder side by feature expansion through upsampling. Here the model learns the unique distribution in real images. 

![encoderstrcutre](imgs/image.png)

# Input And Output From The Encoder Structure

![ioencoder](imgs/image1.png)

# Observation
It is obvious that the autoencoder removes the visible periodic patterns seen in the sample fingervein image with minimal distortion in . So this is consistent with the assumption that these artifacts live in the high frequencies. And intuitively a low pass filtering (a non-linear lowpass in our case), suppressing these noise should make the sythetic samples look like real samples.

To verify this, the fourier features from the denoised samples from the PLUS, SCUT and IDIAP datasets are used to train our PAD system which evaluates the effect of augmentation of real Presentation Attack Instruments( real PAIs) and with synthetic Presentation Attack Instruments(syn PAIs), and also the complete replacement of these real PAIs with syn PAIs. 