# WallCrack_pet_project
Cracked walls classification using unsupervised approaches.
This project combined the approaches of classical clustering methods and unsupervised deep learning.
[Concrete Crack Images for Classification](#CCC) dataset was used. See [this](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

More detailed report on the [link](https://docs.google.com/document/d/1MCauhV5hhBF0vu1yclQCtKX03kNLUnXgQDsQNn8lhKM/edit?usp=sharing)



## Setup data
Before you start, specify the path to the dataset and model:  open **config.py**, change the **PATH** and **model_name** .


## Visualization
If you want to keep track of training and validation losses: open **config.py**, make **train_val_visual = True** <br/>
Figure will be saved and update automatically in **outputs** with name format "**model_name Train Validation loss.png**"<br/>
To plot heatmaps use:<br/>
```bash
python inference.py heatmap_plot N
```
where N - images number. Figures will be stored in **outputs/heat_map/**<br/>
To display the loss distribution use:<br/>
```bash
python inference.py loss_distrib
```
## Train model
```bash
python main.py train
```

## Test
```bash
python inference.py test
```


## Citations
<a id = 'CCC'>
[1] ?zgenel, ?a?lar F?rat (2019), “Concrete Crack Images for Classification”, Mendeley Data, V2, doi: 10.17632/5y9wdsg2zt.2
<br>