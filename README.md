# WallCrack_pet_project
Сracked walls classification using unsupervised approaches.
This project combined the approaches of classical clustering methods and unsupervised deep learning.
[Concrete Crack Images for Classification](#CCC) dataset was used. See [this](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

More detailed report on the [link](https://docs.google.com/document/d/1MCauhV5hhBF0vu1yclQCtKX03kNLUnXgQDsQNn8lhKM/edit?usp=sharing)



## Setup data
Before you start, specify the path to the dataset and model:  open **config.py**, change the **PATH** and **model_name** .

## Train model
```bash
python main.py train
```

## Test
```bash
python main.py test
```


## Citations
<a id = 'CCC'>
[1] Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, V2, doi: 10.17632/5y9wdsg2zt.2
<br>
