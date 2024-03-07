#Load Necessary packages
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn
from PIL import Image
import cv2
import imblearn
from sklearn.preprocessing import StandardScaler

#Load models
class_bb_model = pd.read_pickle('E:\\Winnie Documents\\Guvi\\project\\Copper Modelling\\balancebagging_model.pkl')
reg_rf_model = pd.read_pickle('E:\\Winnie Documents\\Guvi\\project\\Copper Modelling\\random_forest_model.pkl')


#Label Encoding user input
def label_encode(input_type,user_input):
     if input_type == 'country':
          values = {'107': 0,'113': 1,'25': 2,'26': 3,'27': 4,'28': 5,'30': 6,'32': 7,'38': 8,'39': 9,'40': 10,'77': 11,'78': 12,'79': 13,'80': 14,'84': 15,'89': 16}
          res = values[user_input]
     if input_type == 'item':
          values = {'IPL': 0, 'Others': 1, 'PL': 2, 'S': 3, 'SLAWR': 4, 'W': 5, 'WI': 6}
          res = values[user_input]
     if input_type == "application":
          values = {'10': 0, '15': 1, '19': 2, '2': 3, '20': 4, '22': 5, '25': 6, '26': 7, '27': 8, '28': 9, '29': 10, '3': 11, '38': 12, '39': 13, '4': 14, '40': 15, '41': 16, '42': 17, '5': 18, '56': 19, '58': 20, '59': 21, '65': 22, '66': 23, '67': 24, '68': 25, '69': 26, '70': 27, '79': 28, '99': 29}
          res = values[user_input]
     if input_type == 'product':
          values = {'1282007633': 0, '1332077137': 1, '164141591': 2, '164336407': 3, '164337175': 4, '1665572032': 5,'1665572374': 6, '1665584320': 7, '1665584642': 8, '1668701376': 9, '1668701698': 10, '1668701718': 11, '1668701725': 12, 
                   '1670798778': 13, '1671863738': 14, '1671876026': 15, '1690738206': 16, '1690738219': 17, '1693867550': 18, '1693867563': 19,'1721130331': 20, '1722207579': 21, '611728': 22, '611733': 23, '611993': 24, '628112': 25, '628117': 26, '628377': 27,
                   '640400': 28, '640405': 29, '640665': 30, '929423819': 31}
          res = values[user_input]
     if input_type == 'month':
          values = {'January' : [1,0,0,0,0,0,0,0,0,0],'February' : [0,1,0,0,0,0,0,0,0,0],'March': [0,0,1,0,0,0,0,0,0,0],'April' : [0,0,0,1,0,0,0,0,0,0],
                   'May' : [0,0,0,0,0,0,0,0,0,0],'June' : [0,0,0,0,0,0,0,0,0,0],'July' : [0,0,0,0,1,0,0,0,0,0] ,'August' :[0,0,0,0,0,1,0,0,0,0],
                   'September' : [0,0,0,0,0,0,1,0,0,0],'October' :  [0,0,0,0,0,0,0,1,0,0],'November' : [0,0,0,0,0,0,0,0,1,0],'December' : [0,0,0,0,0,0,0,0,0,1]}
          res = values[user_input]
     if input_type == "product_regr":
          values = {'1282007633': 0, '1332077137': 1, '164141591': 2, '164336407': 3, '164337175': 4, '1665572032': 5, '1665572374': 6, '1665584320': 7, '1665584642': 8,
                    '1665584662': 9, '1668701376': 10, '1668701698': 11, '1668701718': 12, '1668701725': 13, '1670798778': 14, '1671863738': 15, '1671876026': 16, 
                    '1690738206': 17, '1690738219': 18, '1693867550': 19, '1693867563': 20, '1721130331': 21,'1722207579': 22, '611728': 23, '611733': 24, '611993': 25,
                      '628112': 26, '628117': 27, '628377': 28, '640400': 29, '640405': 30, '640665': 31,'929423819': 32}
          res = values[user_input]
     return res   
      
if 'pred_price' not in st.session_state:
    st.session_state['pred_price'] = 'False'

if 'pred_status' not in st.session_state:
    st.session_state['pred_status'] = 'False'



#Streamlit Configuration
st.set_page_config(page_title="Copper Industry Sales price / Transaction status predictor", layout="wide", initial_sidebar_state="collapsed", menu_items=None)
selected = option_menu(None, ["Home",'Sales price Prediction','Status prediction'], icons=['house', 'graph-up', 'clipboard-check'], menu_icon="cast", default_index=0, 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#DAA06D"},
        "icon": {"color": "red", "font-size": "20px"}, 
        "nav-link": {"font-color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#A52A2A"}
    })  


if selected == "Home":
    st.markdown("<h2 style='text-align: center; color: Black;'> Copper Industry Sales price / Transaction status predictor</h2>", unsafe_allow_html=True)
    st.markdown('''<p style= 'text-align: center;'>Copper Industrial Sales price and Transaction predictor app helps you to predicts Copper Sales price and Transaction status based 
                on historical data from 2020 to 2021.</p>''', unsafe_allow_html=True)
    image = cv2.imread("E:\\Winnie Documents\\Guvi\\project\\Copper Modelling\\copper_price_chart.png")
    col1,col2,col3 = st.columns([1,1,1])
    with col2:
          st.markdown("<p style= 'text-align: center;'>Historical price trend of copper are as follows,</p>",unsafe_allow_html=True)
          st.image(image)
          st.markdown('''<div style="text-align: justify;">Mean Absolute Error for Selling price prediction is +/- 46.03 and Accuracy for status prediction is ~90% i.e., error rate is 10%.</div>''', unsafe_allow_html=True)
          st.markdown('''<div style="text-align: justify;">**Disclaimer : This project/article is not intended to provide financial, business and investment advice.
                      Users should conduct their due diligence before making any investment decisions using the application</div>''', unsafe_allow_html=True)
          
if selected == "Sales price Prediction":
     st.markdown("<h4 style='text-align: center; color: Black;'> Copper Sales Price Predictor</h4>", unsafe_allow_html=True)
     with st.form("vcform", border=True):
          col1,col2 = st.columns([1,1])
          with col1:
               country_list = ['25', '26', '27', '28', '30', '32', '38', '39', '40', '77', '78', '79', '80', '84', '89', '107', '113']
               country = st.selectbox("Select country",options=country_list)             
               item_type = ['IPL', 'Others', 'PL', 'S', 'SLAWR', 'W', 'WI']             
               item = st.selectbox("Select item type",options=item_type,index=0)
               appl_list = ['2', '3', '4', '5', '10', '15', '19', '20', '22', '25', '26', '27', '28', '29', '38', '39', '40', '41', '42', '56', '58', '59', '65', '66', '67', '68', '69', '70', '79', '99']
               application = st.selectbox("Select application type",options=appl_list)          
               prod_list =['611728', '611733', '611993', '628112', '628117', '628377', '640400', '640405', '640665', '164141591', '164336407', '164337175', '929423819', '1282007633', '1332077137', '1665572032', 
                           '1665572374', '1665584320', '1665584642', '1665584662', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', 
                           '1693867563', '1721130331', '1722207579']
               product = st.selectbox("Select product type",options=prod_list)
          with col2:
               month_list = ['January','February','March','April','May','June','July','August','September','October','November','December']
               month = st.selectbox("Select month of transaction",options=month_list)
               quant = st.number_input('Insert copper Quantity in tons Ex: 102.48(accepts upto 6 decimals)',placeholder="Enter quantity",format="%.6f")             
               thickness = st.number_input('Insert thickness(upto 2 decimals)',placeholder="Enter thickness",format="%.2f")
               width = st.number_input('Insert width(upto 2 decimals)',placeholder="Enter width",format="%.2f")
               
          col3,col4,col5 = st.columns([3.5,0.5,3.5])
          with col4:
              st.write("")
              submit = st.form_submit_button(" Predict ")
              if submit:
               value1 = label_encode("country",country)
               value2 = label_encode("item",item)
               value3 = label_encode("application",application)
               value4 = label_encode("product_regr",product)
               lt1 = [value1,value2,value3,value4]                  
               lt2 = [quant,thickness,width]
               lt3 = label_encode("month",month)
               input_val = lt1+lt2+lt3
               print(input_val)
               with open("E:\Winnie Documents\Guvi\project\Copper Modelling\standard_scalar_regr.pkl",'rb') as f:
                    scr = pickle.load(f)   
               input_arr = np.array([input_val])
               input_sc = scr.transform(input_arr)       
               st.session_state.pred_price = reg_rf_model.predict(input_sc)
               with col5:
                    st.write('') 
                    st.write("Predicted Selling price is",st.session_state.pred_price[0])         
               
          
if selected == "Status prediction":
     st.markdown("<h4 style='text-align: center; color: Black;'> Copper Transaction status predictor</h4>", unsafe_allow_html=True)
     with st.form("vcform", border=True):
          col1,col2 = st.columns([1,1])
          with col1:
               country_list = ['25', '26', '27', '28', '30', '32', '38', '39', '40', '77', '78', '79', '80', '84', '89', '107', '113']
               country = st.selectbox("Select country",options=country_list)             
               item_type = ['IPL', 'Others', 'PL', 'S', 'SLAWR', 'W', 'WI']             
               item = st.selectbox("Select item type",options=item_type,index=0)
               appl_list = ['2', '3', '4', '5', '10', '15', '19', '20', '22', '25', '26', '27', '28', '29', '38', '39', '40', '41', '42', '56', '58', '59', '65', '66', '67', '68', '69', '70', '79', '99']
               application = st.selectbox("Select application type",options=appl_list)          
               prod_list =['611728', '611733', '611993', '628112', '628117', '628377', '640400', '640405', '640665', '164141591', '164336407', '164337175', '929423819', '1282007633', '1332077137', '1665572032', '1665572374', '1665584320', 
                              '1665584642', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
               product = st.selectbox("Select product type",options=prod_list)
               month_list = ['January','February','March','April','May','June','July','August','September','October','November','December']
               month = st.selectbox("Select month of transaction",options=month_list)   
          with col2:
               quant = st.number_input('Insert copper Quantity in tons Ex: 102.48(accepts upto 6 decimals)',placeholder="Enter quantity",format="%.6f")             
               thickness = st.number_input('Insert thickness(upto 2 decimals)',placeholder="Enter thickness",format="%.2f")
               width = st.number_input('Insert width(upto 2 decimals)',placeholder="Enter width",format="%.2f")
               selling_price = st.number_input('Insert Selling price(upto 2 decimals)',placeholder="Enter price",format="%.2f")
          col3,col4,col5 = st.columns([3.5,0.5,3.5])
          with col4:
              st.write("")
              submit = st.form_submit_button(" Predict ")
              if submit:
               val1 = label_encode("country",country)
               val2 = label_encode("item",item)
               val3 = label_encode("application",application)
               val4 = label_encode("product",product)
               l1 = [val1,val2,val3,val4]                  
               l2 = [quant,thickness,width,selling_price]
               l3 = label_encode("month",month)
               input = l1+l2+l3
               print(input)
               with open('E:\Winnie Documents\Guvi\project\Copper Modelling\standard_scalar.pkl','rb') as f:
                    sc = pickle.load(f)   
               input_array = np.array([input])
               input_scaled = sc.transform(input_array)       
               st.session_state.pred_status = class_bb_model.predict(input_scaled) 
               with col5:
                    if st.session_state.pred_status == 1:
                         st.write("")
                         st.markdown("Predicted Status is : Win")
                    if st.session_state.pred_status ==0:
                         st.write("")
                         st.write("Predicted Status is Loss")    