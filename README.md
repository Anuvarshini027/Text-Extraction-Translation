# Text-Extraction-Translation

This app has been deployed on streamlit. To view the app check the link below <br>
https://share.streamlit.io/anuvarshini027/text-extraction-translation/main/text_extraction_translation.py

The corresponding .py file is text_extraction_translation.py which is Helsinski model(MarianMTModel)

## There is another .py file called img_text_translation_mbat.py in which "facebook/mbart-large-50-many-to-many-mmt" model is used. <br>
## But, due to the memory consumed by the model exceeded the maximum limit, deploying in cloud failed.<br>
## So, to use this app, you can download the file and the font files in the same directore and run the command given below.

To run the app in anaconda prompt, go to the location where the App_Higgs_Boson.py file is using the cd command and then run the following line:
```
streamlit run App_Higgs_Boson.py
```
## Streamlit web app implementation of the project. 

## Pre-requisites :

Make sure to install streamlit if haven't already, to install streamlit use the following command :

```
pip install streamlit
```
All the package requirements have been mentioned in the requirements.txt file. 

## Using Anaconda Prompt:

To run the app in anaconda prompt, go to the location where the App_Higgs_Boson.py file is using the cd command and then run the following line:

```
streamlit run App_Higgs_Boson.py
```
## Step-by-Step Approach:

 -Text Extraction from Image using easyocr<br>
 -Inpainting the Image(so that the translated text can be replaced)<br>
 -Translating the extracted text(using hugging-face transformers)<br>
 -Printing translated text back on Image <br>



