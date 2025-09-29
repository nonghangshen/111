
## Experiment
### Requirements
In order to run the project please install the environment by following these commands: 
```
conda create -n TMamba python=3.10
pip install -r requirements.txt
conda activate TMamba
```

### Evaluation 
[Pretrained models] ([https://drive.google.com/drive/folders/1pVhJFwk2f3arP7zUDFAe5_PJrPSG1gc2?usp=drive_link](https://drive.google.com/drive/folders/1Zvkspmz5HiCT3M5_BgYbDlKljtgcOYcj?usp=drive_link)) <br> 


### Testing
```
python main.py --content_dir ./data/cnt --style_dir ./data/sty --mode test
```

### Training  
Style dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
content dataset is COCO2014  <br>  

Here is a partial display of the Chinese painting dataset(https://drive.google.com/drive/folders/1tLjfgiUdwbmtJSaxW4ui05acRNTakTZ6?usp=drive_link)

```
python main.py --content_dir ./data/cnt --style_dir ./data/sty --mode train
```


