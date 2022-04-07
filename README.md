# tabular-synthetic-data
 Synthetic data publishing using generative models
 Each algorithm consists of:
 1. Training function: function which used to train the model
 2. Example architecture for the COVID-19 data case
 3. Example of usage
 
 There are several scripts needed to notice:
 1. data_prep: to call the data
 2. calculate: contains the calculation functions (e.g. KL Divergence using various distributions)

## References
    VAE: D.P. Kingma, M. Welling, Auto-Encoding Variational Bayes, 2nd Int. Conf. Learn. Represent. ICLR 2014 - Conf. Track Proc. (2013). https://arxiv.org/abs/1312.6114v10.
    GAN: I.J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative Adversarial Networks, ArXiv. (2014). https://arxiv.org/abs/1406.2661.
    Wasserstein addition: https://agustinus.kristia.de/techblog/2017/02/04/wasserstein-gan/
    for AAE and AVB adaptation: https://github.com/wiseodd/generative-models