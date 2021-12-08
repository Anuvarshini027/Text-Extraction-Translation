# Text-Extraction-Translation

### This app has been deployed on streamlit. To view the app check the link below <br>
https://share.streamlit.io/anuvarshini027/text-extraction-translation/main/text_extraction_translation.py<br>
### The corresponding .py file is text_extraction_translation.py which is implemented using "Helsinki-NLP/opus-mt" Model.

#### NOTE: There is another .py file called img_text_translation_mbart.py in which "facebook/mbart-large-50-many-to-many-mmt" Model is used.But, due to the memory consumed by the model exceeded the maximum limit, deploying in cloud failed.<br>
#### Thus, to use this app, you can download the file and the font files in the same directory and run the command given below.

To run the app in anaconda prompt, go to the location where the img_text_translation_mbart.py file is using the cd command and then run the following line:<br>
NOTE:Make sure that .ttf files and .py file belongs to the same directory
```
streamlit run img_text_translation_mbart.py
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
streamlit run img_text_translation_mbart.py
```
## Step-by-Step Approach:

 -Text Extraction from Image using easyocr<br>
 -Inpainting the Image(so that the translated text can be replaced)<br>
 -Translating the extracted text(using hugging-face transformers)<br>
 -Printing translated text back on Image <br>



