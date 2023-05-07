import pickle
import cv2
import numpy as np
import mahotas
model=pickle.load(open('svm_model_all.pkl','rb'))
scaler=pickle.load(open('scaler_all.pkl','rb'))
labelenc=pickle.load(open('leenc_all.pkl','rb'))

print(labelenc.classes_)
usedict={
'Alpinia Galanga (Rasna)':'Rasna plant is used in many Ayurvedic medicines in India, Tibet, Africa to help with inflammation, bronchitis, asthma, cough, indigestion, piles, joint pains,obesity, diabetes. The paste of the leaf is also applied externally to reduce swelling. There are many varieties of rasna available throughout India',
'Amaranthus Viridis (Arive-Dantu)':'Amaranthus viridis is used as traditional medicine in the treatment of fever, pain, asthma, diabetes, dysentery, urinary disorders, liver disorders, eye disorders and venereal diseases. The plant also possesses anti-microbial properties',
'Artocarpus Heterophyllus (Jackfruit)':'Jackfruit (Artocarpus heterophyllus Lam) is a rich source of several high-value compounds with potential beneficial physiological activities. It is well known for its antibacterial, antifungal, antidiabetic, anti-inflammatory, and antioxidant activities',
'Azadirachta Indica (Neem)':'Neem is a rich source of limonoids that are endowed with potent medicinal properties predominantly antioxidant, anti-inflammatory, and anticancer activities',
'Basella Alba (Basale)': 'Basella alba is reported to improve testosterone levels in males, thus boosting libido. Decoction of the leaves is recommended as a safe laxative in pregnant women and children. Externally, the mucilaginous leaf is crushed and applied in urticaria, burns and scalds.',
'Brassica Juncea (Indian Mustard)':'It is a folk remedy for arthritis, foot ache, lumbago and rheumatism. Brassica juncea is grown mainly for its seed used in the fabrication of brown mustard or for the extraction of vegetable oil. Brown mustard oil is used against skin eruptions and ulcers',
'Carissa Carandas (Karanda)':'Its fruit is used in the ancient Indian herbal system of medicine, Ayurvedic, to treat acidity, indigestion, fresh and infected wounds, skin diseases, urinary disorders and diabetic ulcer, as well as biliousness, stomach pain, constipation, anemia, skin conditions, anorexia and insanity',
'Citrus Limon (Lemon)':'Aside from being rich in vitamin C, which assists in warding off infections, the juice is traditionally used to treat scurvy, sore throats, fevers, rheumatism, high blood pressure, and chest pain',
'Ficus Auriculata (Roxburgh fig)':'The fruits are edible and are used to make jams, juices and curries in India. In Vietnam, unripe fruits are also used in salads. Leaves are used as fodder for ruminants',
'Ficus Religiosa (Peepal Tree)':'Ficus religiosa, commonly known as pepal belonging to the family Moraceae, is used traditionally as antiulcer, antibacterial, antidiabetic, in the treatment of gonorrhea and skin diseases',
'Hibiscus Rosa-sinensis':'Hibiscus rosa-sinensis is a flowering plant native to tropical Asia. Hibiscus is commonly consumed in teas made from its flowers, leaves, and roots. In addition to casual consumption, Hibiscus is also used as an herbal medicine to treat hypertension, cholesterol production, and cancer progression',
'Jasminum (Jasmine)':'Jasmine is inhaled to improve mood, reduce stress, and reduce food cravings. In foods, jasmine is used to flavor beverages, frozen dairy desserts, candy, baked goods, gelatins, and puddings. In manufacturing, jasmine is used to add fragrance to creams, lotions, and perfumes',
'Mangifera Indica (Mango)':'Various parts of plant are used as a dentrifrice, antiseptic, astringent, diaphoretic, stomachic, vermifuge, tonic, laxative and diuretic and to treat diarrhea, dysentery, anaemia, asthma, bronchitis, cough, hypertension, insomnia, rheumatism, toothache, leucorrhoea, haemorrhage and piles',
'Mentha (Mint)':'Mentha species are widely used in savory dishes, food, beverages, and confectionary products. Phytochemicals derived from mint also showed anticancer activity against different types of human cancers such as cervix, lung, breast and many others',
'Moringa Oleifera (Drumstick)':'Moringa supplies a good source of vitamin C, an antioxidant nutrient that supports immune function and collagen production',
'Muntingia Calabura (Jamaica Cherry-Gasagase)':'Antiseptic properties and therapeutic uses of the flowers include the treatment of abdominal cramps and spasms',
'Murraya Koenigii (Curry)':'They are used as antihelminthics, analgesics, digestives, and appetizers in Indian cookery . The green leaves of M. koenigii are used in treating piles, inflammation, itching, fresh cuts, dysentery, bruises, and edema. The roots are purgative to some extent',
'Nerium Oleander (Oleander)':'Anvirze is an aqueous extract of the plant Nerium oleander which has been utilized to treat patients with advanced malignancies . Other medicinal uses of Nerium oleander include treating ulcers, haemorrhoids, leprosy, to treat ringworm, herpes, and abscesses',
'Nyctanthes Arbor-tristis (Parijata)':'The leaves of the Nyctanthes arbor-tristis plant find their use in Ayurveda and Homoeopathy for the treatment of sciatica, fevers, and arthritis. They are also used as a laxative for treating constipation. The plant has properties that help treat snake bites',
'Ocimum Tenuiflorum (Tulsi)':'Tulsi has also been shown to counter metabolic stress through normalization of blood glucose, blood pressure and lipid levels, and psychological stress through positive effects on memory and cognitive function and through its anxiolytic and anti-depressant properties',
'Piper Betle (Betel)':'Traditionally, the plant is used to cure many ailments such as cold, bronchial asthma, cough, stomachalgia and rheumatism',
'Plectranthus Amboinicus (Mexican Mint)':'It is widely used in folk medicine to treat conditions like cold, asthma, constipation, headache, cough, fever and skin diseases. The leaves of the plant are often eaten raw or used as flavoring agents, or incorporated as ingredients in the preparation of traditional food',
'Pongamia Pinnata (Indian Beech)':'The Indian beech, Pongam seed oil tree or Hongay seed oil, is a medium-sized, glabrous, semi-evergreen tree. The fruits, bark, seeds, seed oil, leaves, roots and flowers of Pongamia pinnata have all been recommended for analgesic, arthritis and inflammatory activity',
'Psidium Guajava (Guava)':'Although guava has a number of medicinal properties, it is the most common and popular traditional remedy for gastrointestinal infections such as diarrhea, dysentery, stomach aches, and indigestion and it is used across the world for these ailments',
'Punica Granatum (Pomegranate)':'The pomegranate polyphenol; punicalagin, is known to have potent anticancer activity in breast, lung, and cervical cells. All parts of the fruit were reported to have therapeutic activity including anticancer, anti-inflammatory, anti-atherogenic, anti-diabetes, hepato protective, and antioxidant activity, etc',
'Santalum Album (Sandalwood)':'Sandalwood has antipyretic, antiseptic, antiscabetic, and diuretic properties. It is also effective in treatment of bronchitis, cystitis, dysuria, and diseases of the urinary tract',
'Syzygium Cumini (Jamun)':'The bark is acrid, sweet, digestive, astringent to the bowels, anthelmintic and used for the treatment of sore throat, bronchitis, asthma, thirst, biliousness, dysentery and ulcers. It is also a good blood purifier',
'Syzygium Jambos (Rose Apple)':'Rose apple has a long history of being used in traditional and folk medicine in various cultures. In the Chinese system of traditional medicine, the fruit and root bark are believed to be of use as a blood coolant. The fruit has been used as a diuretic and as a tonic for better health of the brain and liver',
'Tabernaemontana Divaricata (Crape Jasmine)':'Tabernaemontana divaricata has several uses in medicine. In Ayurvedic medicine, the juice from the flower buds is mixed with oil and applied to the skin to treat inflammation. It is also used in dental care, for scabies, as cough medicine and for eye ailments',
'Trigonella Foenum-graecum (Fenugreek)':'Fenugreek is a herb that is widely used in cooking and as a traditional medicine for diabetes in Asia. It has been shown to acutely lower postprandial glucose levels, but the long-term effect on glycemia remains uncertain'


}

print(usedict['Amaranthus Viridis (Arive-Dantu)'])


# Converting each image to RGB from BGR format
bins= 8
def Convert_to_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img


# Conversion to HSV image format from RGB

def Convert_to_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img



# image segmentation

# for extraction of green and brown color


def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result




# feature-descriptor-1: Hu Moments
def get_shape_feats(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
# feature-descriptor-2: Haralick Texture
def get_texture_feats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
# feature-descriptor-3: Color Histogram
def get_color_feats(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def test_leaf(impath):
    image = cv2.imread(impath)
    image = cv2.resize(image, (500, 500))

    bgrim       = Convert_to_bgr(image)
    
    hsvim       = Convert_to_hsv(bgrim)
  
    seg_image   = img_segmentation(bgrim,hsvim)


    f_shape = get_shape_feats(seg_image)
    f_text   = get_texture_feats(seg_image)
    f_color  = get_color_feats(seg_image)

    # Concatenate 

    f_combined = np.hstack([f_color, f_text, f_shape])
    
    # fdata=np.array([f_combined])
    print(f_combined)
    fdata=scaler.transform([f_combined])
    print(fdata)

    ypred=model.predict(fdata)
    print("ypred==>",ypred)
    label=labelenc.inverse_transform(ypred)
    disname=label[0]
    print("disease==>",disname)
    try:
        usval=usedict[disname]
    except:

        usval='Not a medicinal plant'
    print("use===>",usval)
    return disname,usval






if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    test_leaf(path)
   


# imp="dataset/Pepper__bell___healthy/0a3f2927-4410-46a3-bfda-5f4769a5aaf8___JR_HL 8275.jpg"
# test_leaf(imp)
