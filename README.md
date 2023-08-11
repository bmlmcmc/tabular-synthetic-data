# tabular-synthetic-data
 Synthetic data publishing using generative models for tabular data
 Each algorithm consists of:
 1. Training function: function which used to train the model
 2. Example architecture for the COVID-19 data case
 3. Example of usage

 There are several scripts needed to notice:
 1. data_prep: to call the data
 2. calculate: contains the calculation functions (e.g. KL Divergence using various distributions)

This code was used to create the paper titled [Data Analysis and Synthesis of COVID-19 Patients using Deep Generative Models: A Case Study of Jakarta, Indonesia](https://ieeexplore.ieee.org/document/9921948/). If you use this code, please cite our paper as well here:
```
<h1>B. I. Nasution, I. D. Bhaswara, Y. Nugraha and J. I. Kanggrawan, "Data Analysis and Synthesis of COVID-19 Patients using Deep Generative Models: A Case Study of Jakarta, Indonesia," 2022 IEEE International Smart Cities Conference (ISC2), Pafos, Cyprus, 2022, pp. 1-7, doi: 10.1109/ISC255366.2022.9921948.</h1>
```
Or using the bibtex:
```
@INPROCEEDINGS{9921948,
  author={Nasution, Bahrul Ilmi and Bhaswara, Irfan Dwiki and Nugraha, Yudhistira and Kanggrawan, Juan Intan},
  booktitle={2022 IEEE International Smart Cities Conference (ISC2)}, 
  title={Data Analysis and Synthesis of COVID-19 Patients using Deep Generative Models: A Case Study of Jakarta, Indonesia}, 
  year={2022},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/ISC255366.2022.9921948}}
```

Special thanks to: [Irfan Dwiki Bhaswara](https://github.com/bhaswara) who helped in construction of the scripts

## References
    VAE: D.P. Kingma, M. Welling, Auto-Encoding Variational Bayes, 2nd Int. Conf. Learn. Represent. ICLR 2014 - Conf. Track Proc. (2013). https://arxiv.org/abs/1312.6114v10.
    GAN: I.J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative Adversarial Networks, ArXiv. (2014). https://arxiv.org/abs/1406.2661.
    AAE: A. Makhzani, J. Shlens, N. Jaitly and I. Goodfellow, "Adversarial autoencoders", International Conference on Learning Representations, (2016). http://arxiv.org/abs/1511.05644.
    AVB: L. Mescheder, S. Nowozin and A. Geiger, "Adversarial variational bayes: Unifying variational autoencoders and generative adversarial networks", Proceedings of the 34th International Conference on Machine Learning - Volume 70 ser. ICML'17, pp. 2391-2400, (2017). https://arxiv.org/abs/1701.04722.
    Wasserstein distance: M. Arjovsky, S. Chintala and L. Bottou, "Wasserstein Generative Adversarial Networks" in Proceedings of the 34th International Conference on Machine Learning ser. Proceedings of Machine Learning Research, vol. 70, pp. 214-223, (2017). https://proceedings.mlr.press/v70/arjovsky17a.html

    [AAE and AVB implementation adaptation](https://github.com/wiseodd/generative-models)
    [Wasserstein addition](https://agustinus.kristia.de/techblog/2017/02/04/wasserstein-gan/)
