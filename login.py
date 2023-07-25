import streamlit as st
import streamlit_authenticator as stauth
# import BigBooks
from  main import myMain
# import yaml
# from yaml.loader import SafeLoader
#
# with open(r'D:\lvshaomei\modelDeploy\yolov5-streamlit-main\yolov5-streamlit-main\config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

st.set_page_config(page_title="模型部署", page_icon="🎈", layout="wide")

# 如下代码数据，可以来自数据库
names = ['吕少梅', '管理员']
usernames = ['lvshaomei', 'admin']
passwords = ['S0451', 'ad4516']


hashed_passwords = stauth.Hasher(passwords).generate()


credentials = {'usernames': {
                usernames[0]: {'email': 'xiaoyw****@gmail.com',
                            'name': names[0],
                            'password': hashed_passwords[0]},
                usernames[1]: {'email': 'admin***@gmail.com',
                            'name': names[1],
                            'password': hashed_passwords[1]}
                            }
               }

authenticator = stauth.Authenticate(credentials,
    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)




name, authentication_status, username = authenticator.login('Login', 'main')
print(authentication_status)

if authentication_status:
    with st.container():
        cols1,cols2 = st.columns(2)
        cols1.write('欢迎 *%s*' % (name))
        with cols2.container():
            authenticator.logout('Logout', 'main')

    myMain()
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
