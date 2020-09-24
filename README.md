# graphvae_compress
Tensorflow implementation of the model described in the paper [Compressed Graph Representation for Scalable Molecular Graph Generation](https://doi.org/10.1186/s13321-020-00463-2)

## Components
- **preprocessing.py** - script for data preprocessing
- **train.py** - script for model training
- **test.py** - script for model evaluation (molecular graph generation)
- **GVAE.py** - model architecture
- **util.py**

## Dependencies
- **Python**
- **TensorFlow**
- **RDKit**
- **NumPy**
- **scikit-learn**
- **sparse**

## Citation
```
@Article{Kwon2020,
  title={Compressed graph representation for scalable molecular graph generation},
  author={Kwon, Youngchun and Lee, Dongseon and Choi, Youn-Suk and Shin, Kyoham and Kang, Seokho},
  journal={Journal of Cheminformatics},
  volume={12},
  pages={58},
  year={2020},
  doi={10.1186/s13321-020-00463-2}
}
```
