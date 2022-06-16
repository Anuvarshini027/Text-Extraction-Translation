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
## Pre-requisites :

Make sure to install streamlit if haven't already, to install streamlit use the following command :

```
pip install streamlit
```
All the package requirements have been mentioned in the requirements.txt file. 


## Step-by-Step Approach:

 -Text Extraction from Image using easyocr<br>
 -Inpainting the Image(so that the translated text can be replaced)<br>
 -Translating the extracted text(using hugging-face transformers)<br>
 -Printing translated text back on Image <br>

## Output

### Input Image

![image](https://user-images.githubusercontent.com/60288450/158811169-d3545bec-52c8-488b-ad25-6b005b501571.png)

### Detected Text in Image

![image](https://user-images.githubusercontent.com/60288450/158811400-a79a42c0-1dd9-4212-8f43-29b19af53154.png)

### Translated Image

![image](https://user-images.githubusercontent.com/60288450/158811605-415bc24a-4ada-411e-a7cc-d625e425b25d.png)

## Another Example: From English to French

![image](https://user-images.githubusercontent.com/60288450/158812458-fccf997b-22b1-40f2-a5c1-2a1c1c3d10a7.png)

### Translated Image
![image](https://user-images.githubusercontent.com/60288450/158812667-ab9bfd90-d1ab-4c63-a7ac-16597d5c20d9.png)
