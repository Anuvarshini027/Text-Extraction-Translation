
import streamlit as st
st.title("Image Text Extraction and Translation")
import easyocr
from PIL import ImageFont, ImageDraw, Image
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import time


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def inpaint_easyocr(img,reader):
    img_copy = img.copy()
    #plt.imshow(img)
    #plt.show()
    result = reader.readtext(img)
    text = [result[i][1] for i in range(len(result))]
    # text=' '.join(map(str,text))

    kernel = np.ones((1, 1), np.uint8)
    # print(img_copy)
    mask = np.zeros(img.shape[:2], dtype="uint8")

    for i in range(len(result)):
        l = []
        for box in result[i]:
            l.append(box)
            x0, y0 = l[0][0]
            x1, y1 = l[0][1]
            x2, y2 = l[0][2]
            x3, y3 = l[0][3]
            x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
            x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
            thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

            rect = cv2.rectangle(img, (int(x0), int(y0)), ((int(x2), int(y2))),
                                 (0, 255, 0), 2)
            img_copy = cv2.inpaint(img_copy, mask, 7, cv2.INPAINT_NS)
            # img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)
            # img_copy=cv2.dilate(img_copy,kernel,iterations = 1)
    return (img_copy, rect, text, result)

def wrap_text(text, font, max_width):

    lines = []

        # If the text width is smaller than the image width, then no need to split
        # just add it to the line list and return
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        # split the line by spaces to get words
        words = text.split(' ')
        i = 0
            # append every word to a line while its width is shorter than the image width
        while i < len(words):
            line = ''
            while (i < len(words) and font.getsize(line + words[i])[0] <= max_width):
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line)
    return lines


def preprocessing(text):
    if "com" in (text[-1]):
        # to remove .com if it is present at the end(Note: it is assumed that websites will be present at the end generally)
        com = text.pop(-1)
        # print(com)
    text2 = ' '.join(text)

    text1 = re.sub(r'[0-9]+', ' ', text2)
    text1 = re.sub(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff \n / > < '' ' ' "" " " }{][ ]',
        '', text1)

    # removing square brackets
    # Deletes particular pattern
    text1 = re.sub('\[.*?\]', '', text1)

    text1 = re.sub('<.*?>+', '', text1)
    # removing hyperlink
    text1 = re.sub('https?://\S+|www\.\S+', '', text1)

    # removing puncuation
    text1 = re.sub('[%s]' % re.escape(string.punctuation), '', text1)

    text1 = re.sub('\n', '', text1)

    # remove words containing numbers
    text1 = re.sub('\w*\d\w*', '', text1).split()

    text1 = " ".join(text1)
    return text1


def translated_img_other_lang(inpaint_img, result, fontpath,
                              translated_text_str, fontsize):
    img2 = inpaint_img.copy()
    b, g, r = 0, 0, 0
    a, b, c = inpaint_img.shape  # a=height,b=width
    gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    moment = cv2.moments(gray_img)
    x = []
    y = []
    for i in range(len(result)):
        x.append(result[i][0][0][0])  # gives starting x cordinate x1
        x.append(result[i][0][2][0])  # gives ending x cordinate x3
        y.append(result[i][0][0][1])  # gives starting y cordinate y1
        y.append(result[i][0][2][1])  # gives ending y cordinate y3

    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    X = min_x + 5
    Y = min_y + 5
    # X = int(moment ["m10"] / moment["m00"])/4.5
    # Y = int(moment ["m01"] / moment["m00"])/4
    # Y=10

    font = ImageFont.truetype(fontpath, fontsize)
    lines = wrap_text(translated_text_str, font, max_x - min_x)
    print(lines)
    img_pil = Image.fromarray(img2)
    for i in lines:
        draw = ImageDraw.Draw(img_pil)
        draw.text((X, Y), i, font=font, fill=(b, g, r))
        # X=X+len(translated_text_list[i])
        Y = Y + ((a / 10) + 10)  # gives space dynamically
    img2 = np.array(img_pil)

    return (img2)
    
def shape(inpainted):
   if (inpainted.shape[0]) <=400:
       fontsize=25
   elif  (inpainted.shape[0]) >400 and  (inpainted.shape[0]) <=600:
       fontsize=40
   elif  (inpainted.shape[0]) >600 and  (inpainted.shape[0]) <=1000:
       fontsize=60
   elif (inpainted.shape[0]) >1000:
       fontsize=100
   return fontsize
   

st.subheader("Please Wait. Loading...")
model_mBart = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt")
tokenizer_mBart = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt")

st.subheader('Upload any image which needs to be translated')

# if the user chooses to upload the data
file = st.file_uploader('Image file')
# browsing and uploading the dataset (strictly in csv format)
# dataset = pd.DataFrame()


dict_language = {
    "Tamil": "ta",
    "Telugu": "te",
    "Chinese": "ch_sim",
    # "TChinese":"ch_tra",
    "Japanese": "ja",
    "Vietnamese": "vi",
    "Lithuanian": "lt",
    "French": "fr",
    "Thai": "th",
    "Czech": "cs",
    "Bengali": "bn",
    "Spanish": "es",
    "Russian": "ru",
    'Urdu': 'ur',
    'Latvian': 'lv',
    'Latin': 'la',
    'Italian': 'it',
    'Swahili': 'sw',
    'Ukrainian': 'uk',
    'Korean': 'ko',
    'English': 'en',
    'Romanian': 'ro',
    'Nepali': 'ne',
    'Indonesian': 'id',
    'Marathi': 'mr',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'German': 'de',
    'Persian': 'fa',
    'Estonian': 'et',
    'Polish': 'pl',
    'Sweden': 'sv',
    'Norwegian': 'no',
    "Portuguese": "pt",
    "Turkish": "tr",
    "Afrikaans": "af",
    "Dutch": "nl",
    "Tagalog": "tl",
    "Slovenian": "sl"

}

target_lang = {
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovene": "sl_SI"
}

if file is not None:
    my_img = Image.open(file)
    st.image(my_img , caption='Uploaded Image.', width=None)
    
    img_1=np.array(my_img)
    option = st.selectbox('Select the source language',
                          tuple(dict_language.keys()))
    time.sleep(10)
    st.write('You selected:', option)
    
    lang=dict_language[option]

    reader = easyocr.Reader([lang], gpu=False)
    #img_1 = cv2.imread(file)
    inpainted, rect, text, result = inpaint_easyocr(img_1,reader)
    
    st.write(f"The detected text is: {text}")
    st.subheader('Inpainted Image')
    st.image(inpainted)
    #st.image(plt.show())
    st.image(rect)

    target_option = st.selectbox('Select the Target language',
                                 tuple(target_lang.keys()))
    if option == tgt_input:
       st.warning(f"The Original text is already in {option},Consider changing it.")
    time.sleep(10)
    st.write('Your selected target language is :', target_option)
    src= target_lang[option]
    tgt=target_lang[target_option]
    
    text_formatted = preprocessing(text)

    tokenizer_mBart.src_lang = src
    encoded_mBart = tokenizer_mBart(text_formatted, return_tensors="pt")
    generated_tokens = model_mBart.generate(**encoded_mBart,
                                            forced_bos_token_id=
                                            tokenizer_mBart.lang_code_to_id[tgt])
    translated_text = tokenizer_mBart.batch_decode(generated_tokens,
                                                   skip_special_tokens=True)
    translated_text_str = " ".join(translated_text)
    st.write(translated_text_str)

    if tgt =='en_XX':
        fontpath="times-new-roman.ttf"
        f=shape(inpainted)
        trans_img_lang=translated_img_other_lang(inpainted,result,fontpath,translated_text_str,f)
        st.image(trans_img_lang) 

    if tgt ==  "si_LK" :
        fontpath = "Akshar Unicode.ttf"
        f=shape(inpainted)
        trans_img_lang=translated_img_other_lang(inpainted,result,fontpath,translated_text_str,f)
        st.image(trans_img_lang) 

    if tgt == "de_DE":
        fontpath = "German.ttf"
        f=shape(inpainted)
        trans_img_lang=translated_img_other_lang(inpainted,result,fontpath,translated_text_str,f)
        st.image(trans_img_lang) 


    if tgt == "es_XX":
        fontpath = "spanish.ttf"
        f=shape(inpainted)
        trans_img_lang=translated_img_other_lang(inpainted,result,fontpath,translated_text_str,f)
        st.image(trans_img_lang) 

    if tgt ==  "fr_XX":
        fontpath = "French.ttf"
        f=shape(inpainted)
        trans_img_lang=translated_img_other_lang(inpainted,result,fontpath,translated_text_str,f)
        st.image(trans_img_lang) 

    lang_list= ["ja_XX","ko_KR","zh_CN" ,"th_TH" ,"ar_AR" ,"pt_XX" ,"tr_TR","vi_VN","ru_RU","ka_GE","gu_IN","bn_IN" ,"ta_IN", "te_IN" ,"ml_IN" ,"hi_IN"]
    if tgt in lang_list :
        fontpath = "arial-unicode-ms.ttf"
        f=shape(inpainted)
        trans_img_lang=translated_img_other_lang(inpainted,result,fontpath,translated_text_str,f)
        st.image(trans_img_lang) 
    if(st.button("FINISH")):
        st.info("Thank You for your Patience!")
        st.balloons()

else:
    st.warning("No file has been chosen yet")        
