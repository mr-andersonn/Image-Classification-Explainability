# Image-Classification-Explainability

DAT255 Project Group 42

This project implements image classification with explainability methods for fish species recognition, including a web application and FastAPI backend for visualizing model predictions.

## Before we start
Before you start set up the project, set up your development enviroment as you see fit. The development have been done in linux using a .venv. But other options like conda are also possible.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies:
- TensorFlow 2.21.0
- NumPy, Pandas
- OpenCV, Pillow
- Scikit-learn, Scikit-image
- Matplotlib, Seaborn
- KaggleHub
- keras

## "Project_setup" directory guide.
In project setup directory there are notebooks for testing the setup, and checking some of the main requirements are installed correctly. Such as Tensorflow with GPU support.
- `ML/Project_setup/Check_setup.ipynb` - Check if the setup is correct.

In Download dataset notebook, there is a functionality to download the dataset from Kaggle. Set up your own desired paths. Furthermore there is a cell that cleans the datset so it follows the same directory structure as is used in model_creation notebooks.
- `ML/Project_setup/Download_dataset.ipynb` - Download the dataset.

### Dataset

We use the [Large-Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) from Kaggle.

**Citation:**
```
@inproceedings{ulucan2020large,
   title={A Large-Scale Dataset for Fish Segmentation and Classification},
   author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
   booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
   pages={1--5},
   year={2020},
   organization={IEEE}
}
```

## Model Training

The trained model that is used in report and on web application is provided as `MobileNetV3Large_Improved.keras`.
To retrain:
1. Prepare the cleaned dataset
2. Use the notebooks/scripts in `ML/Model_Creation/MobileNetV3Large_Improved.ipynb`
3. If you want to train the other models as well use the notebooks/scripts in `ML/Model_Creation`

## Running Explainability Visualizations

While there are single files for each explainability method that can be run using the files in `ML/Visualization_files`. The main notebook `ML/Visualization_in_one.ipynb` contains all the visualizations, and have been the preffered method during testing.

in `Visualization_in_one.ipynb` you will se that in the second cell you can choose which picture you would like to visualize, or using functionality implemented in `ML/Visualization_files/gather_image.py` you can visualize all the pictures in the dataset.

## Deployment

Running the application locally involves 2 steps:

### 1. Start the FastAPI server
Download the repository and navigate to the following folder:

`Image-Classification-Explainability/ML/FastAPI/`

Start the server with the command:

```bash
uvicorn main:app --reload
```

### 2. Start the ASP.NET MVC application
Navigate to the following folder:

`Image-Classification-Explainability/Web/ICE-Website/`

Run the command:

```bash
dotnet run
```

The link to the website will be printed in the terminal.
