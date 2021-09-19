# Locally Aligned Embeddings for Schema Linking

## Introduction
This project was completed for the Database Systems class taught by Dr. Georgia
Koutrika, as part of the Data Science and Information Technologies MSc programme
at the University of Athens.

Instructions on how to download the necessary files and how to run the code can
be found bellow.
It is also possible to run the experiments on Google Colab, without downloading
anything, by using the notebook in this repository.
To do so click here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geokats/schema-linking-embeddings/blob/main/sl_emb_experiments.ipynb)

## Data

### WikiSQL Dataset
We use the WikiSQL dataset, which has been copied in our repository from the
authors' original repository. To unpack use:

```
tar xvjf data.tar.bz2
```

### Pre-trained Word embeddings
A set of pre-trained word embedding vectors is needed to run the program.
These can be downloaded from the following links:

- [FastText](https://fasttext.cc/docs/en/english-vectors.html)
- [Word2Vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)


## Acknowledgments
This project is built on two earlier works:

- [EmbDI](https://gitlab.eurecom.fr/cappuzzo/embdi) [1]

  ```
  @article{Cappuzzo_2020,
     title={Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks},
     ISBN={9781450367356},
     url={http://dx.doi.org/10.1145/3318464.3389742},
     DOI={10.1145/3318464.3389742},
     journal={Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data},
     publisher={ACM},
     author={Cappuzzo, Riccardo and Papotti, Paolo and Thirumuruganathan, Saravanan},
     year={2020},
     month={May}
  }
  ```

-  [FastText alignment package](https://github.com/facebookresearch/fastText/tree/master/alignment) [2]

  ```
  @InProceedings{joulin2018loss,
      title={Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion},
      author={Joulin, Armand and Bojanowski, Piotr and Mikolov, Tomas and J\'egou, Herv\'e and Grave, Edouard},
      year={2018},
      booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  }
  ```

If you use them in your work, please acknowledge the authors' contributions
by citing them, using the bibtex citations above.
