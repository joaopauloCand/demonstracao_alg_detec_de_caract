import streamlit as st
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison

#st.set_page_config(layout="wide")

st.title("Comparação de Algoritmos de Detecção de Características: SIFT, SURF e ORB")

img_file_buffer = st.camera_input("Foto inicial do objeto ou cena a ser analisada: ")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    #conversao de BGR para RGB
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # Solicitar segunda imagem apenas se a primeira foi inserida
    img_file_buffer_2 = st.camera_input("Foto da segunda imagem ou cena: ")
    
    if img_file_buffer_2 is not None:
        bytes_data_2 = img_file_buffer_2.getvalue()
        cv2_img_2 = cv2.imdecode(np.frombuffer(bytes_data_2, np.uint8), cv2.IMREAD_COLOR)
        cv2_img_2 = cv2.cvtColor(cv2_img_2, cv2.COLOR_BGR2RGB)
        escolha = st.selectbox(
            "Qual algoritmo você gostaria de testar?",
            ("SIFT", "SURF", "ORB"),
            index=None,
            placeholder="Selecione um algoritmo de detecção de características",
        )
        st.write("Algoritmo selecionado:", escolha)
        if escolha:
            algoritmo = None
            norma_match = None

            if escolha == "SIFT": 
                algoritmo = cv2.SIFT_create()
                norma_match = cv2.NORM_L2
            elif escolha == "SURF":
                try:
                    algoritmo = cv2.xfeatures2d.SURF_create()
                    norma_match = cv2.NORM_L2
                except (AttributeError, cv2.error):
                    st.error("SURF não está disponível nesta instalação do OpenCV (patenteado), use ORB/SIFT :(")
            elif escolha == "ORB":
                algoritmo = cv2.ORB_create()
                norma_match = cv2.NORM_HAMMING

            if algoritmo is None:
                st.stop()

            #keypoints e descritores para a primeira imagem
            kp1, des1 = algoritmo.detectAndCompute(cv2_img, None)
            #keypoints e descritores para a segunda imagem
            kp2, des2 = algoritmo.detectAndCompute(cv2_img_2, None)
            
            # Visualização dos keypoints com e sem tamanho para a primeira imagem
            keypoints_com_tamanho_img1 = np.copy(cv2_img)
            keypoitns_sem_tamanho_img_1 = np.copy(cv2_img)
            cv2.drawKeypoints(cv2_img, kp1, keypoints_com_tamanho_img1, color = (255, 0, 0))
            cv2.drawKeypoints(cv2_img, kp1, keypoitns_sem_tamanho_img_1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #Visualização dos keypoints com e sem tamanho para a segunda imagem
            keypoints_com_tamanho_img2 = np.copy(cv2_img_2)
            keypoitns_sem_tamanho_img_2 = np.copy(cv2_img_2)
            cv2.drawKeypoints(cv2_img_2, kp2, keypoints_com_tamanho_img2, color = (255, 0, 0))
            cv2.drawKeypoints(cv2_img_2, kp2, keypoitns_sem_tamanho_img_2, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.write("Comparação entre visualização de keypoints da imagem 1:")
                image_comparison(
                    img1=keypoints_com_tamanho_img1,
                    img2=keypoitns_sem_tamanho_img_1,
                    label1="Apenas keypoints",
                    label2="KeyPoints com tamanho",
                )

            with col2:
                st.write("Comparação entre visualização de keypoints da imagem 2:")
                image_comparison(
                    img1=keypoints_com_tamanho_img2,
                    img2=keypoitns_sem_tamanho_img_2,
                    label1="Apenas keypoints",
                    label2="KeyPoints com tamanho",
                )

            if des1 is None or des2 is None:
                st.warning("Não foi possível calcular descritores em uma das imagens. Tente imagens com mais textura ou contraste.")
            else:
                # Cria um objeto BFMatcher para correspondência de descritores
                bf = cv2.BFMatcher(norma_match, crossCheck = True)
                # Faz a correspondência entre os descritores das duas imagens
                matches = bf.match(des1, des2)
                # Ordena as correspondências com base na distância
                matches = sorted(matches, key = lambda x:x.distance)

                if len(matches) == 0:
                    st.warning("Nenhuma correspondência encontrada entre as duas imagens.")
                else:
                    limite_matches = st.slider(
                        "Quantidade de correspondências que serão exibidas:",
                        min_value=1,
                        max_value=len(matches),
                        value= len(matches)//2,
                        step=1,
                    )
                    matches_filtrados = matches[:limite_matches]
                    resultado = cv2.drawMatches(cv2_img, kp1, cv2_img_2, kp2, matches_filtrados, None, flags = 2)
                    st.write(f"Correspondências entre as duas imagens (top {limite_matches}):")
                    st.image(resultado)
            