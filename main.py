
import cv2 as cv
import numpy as np
import img_function as imf
import some_blur as sb
import canny_edge_detector as ced
import bilateral_filter as bf
import tkinter as tk
import tkinter.messagebox


######################################
######################################
######################################
######## CARTOONIFYING A PHOTO ########
######################################
######################################
######################################


def esegui_codice():


    if text_input2.get() != "" and text_input3.get() != "" and text_input4.get() != "": 
        # 1) CONVERSIONE IN GRAYSCALE E APPLICAZIONE FILTRO MEDIANO 
        print("START!")
        print("\n")
        
        img= cv.imread("input/" + variable.get())
        img_color= img
        
        cv.imshow("input_image", img )
        print("STOP! ASPETTO UN TASTO")
        print("\n")
        
        cv.waitKey(0)
        print("RIPRENDI")
        print("\n")
        
        print(" SONO IN 1! MEDIAN FILTER! ")
        print("\n")
        for i in range(3):
            img[:,:,i] = sb.median_filter(img[:,:,i],7)
        cv.imwrite(r"output\1_median.jpg", img)
        img= imf.rgb2gray(img)


        # 2) APPLICAZIONE DEL CANNY EDGE DETECTOR 

        ## 2.1- riduzione del rumore: applico filtro gaussiano
        print(" SONO IN 2.1! GAUSS FILTER! ")
        print("\n")
        sigma= int(text_input2.get())
        gauss_kernel= ced.gaussian_kernel(sigma)
        blurred_image= imf.convolve(img, gauss_kernel, True)
        cv.imwrite(r"output\2_1_gaussed_img.jpg", blurred_image)

        ## 2.2- calcolo del gradiente con i filtri di Sobel
        print(" SONO IN 2.2! SOBEL FILTER! ")
        print("\n")
        Mag,Pha= ced.sobel_filters(blurred_image) 
        cv.imwrite(r"output\2_2_Mag.jpg",Mag)

        ## 2.3- assottiglio i bordi con la NMS (Non Maximum Suppression)
        print(" SONO IN 2.3! NON-MAXIMUM SUPPRESSION! ")
        print("\n")
        nms= ced.non_max_suppression(Mag,Pha)
        cv.imwrite(r"output\2_3_NMS.jpg",nms)

        ## 2.4- Thresholding: 
        ## divido i pixels della foto in 3 categorie in base alla loro luminosità: 
        ## strong, weak, 0
        print(" SONO IN 2.4! THRESHOLDING! ")
        print("\n")
        #0,01*255 oppure 2.55 
        lowThresholdRatio= float(text_input3.get())*255
        #0,10*255 oppure 25.5
        highThresholdRatio= float(text_input4.get())*255
        weak=np.int32(80)
        strong=np.int32(255)
        thr= ced.threshold(nms,lowThresholdRatio ,highThresholdRatio , strong, weak)
        cv.imwrite(r"output\2_4_THRESHOLD.jpg",thr)

        ## 2.5- Hysteresis
        ## divido i pixels in due categorie:
        ## strong, 0
        ## un pixel diventa strong se o è già strong oppure se è weak ed ha uno strong intorno. 0 altrimenti
        print(" SONO IN 2.5! HYSTERESIS! ")
        print("\n")
        hys_img= ced.hysteresis(thr, weak, strong)
        cv.imwrite(r"output\2_5_HYSTERESIS.jpg", hys_img)


        # 3) MORPHOLOGICAL OPERATIONS: DILATION
        ## se trovo un pixel luminoso(255) 
        ## rendo luminosi i 3 pixel intorno ad esso
        ## kernel di 2x2. Il pixel che controllo è in posizione [0,0] del kernel
        print(" SONO IN 3! DILATION! ")
        print("\n")
        dilation_img = imf.dilation_2x2(hys_img)
        cv.imwrite(r"output/3_DILATION.jpg", dilation_img)


        # 4) COLOR IMAGE: BILATERAL FILTER
        ## applico  il bilateral filter 14 volte
        ## per ottenere effetto "cartoon"

        ## 4.1- Down_Sampled per ridurre le dimensioni dell' immagine:
        ## per velocizzare il bilateral filter
        ## scompongo l'immagine nei 3 canali
        ## immagine ridotta di un fattore di 4 (col//4,row//4)
        print(" SONO IN 4.1! BILATERAL FILTER! ")
        print("\n")
        B,G,R= cv.split(img_color)
        s_B= imf.down_sampled_one_channel(B, 4)
        cv.imwrite(r"output/4_1_B_down_sampled.jpg", s_B)
        s_G= imf.down_sampled_one_channel(G, 4)
        cv.imwrite(r"output/4_2_G_down_sampled.jpg", s_G)
        s_R= imf.down_sampled_one_channel(R, 4)
        cv.imwrite(r"output/4_3_R_down_sampled.jpg", s_R)

        ## 4.2- Applico bilateral_filter per ogni canale:
        ## kernel 9x9
        ## sigma_color= 17
        ## sigma_space= 17
        print(" SONO IN 4.2.B! ")
        print("\n")
        for i in range(14):
            res_B= bf.bil_filter(s_B, 9,17,17)
        cv.imwrite(r"output/4_4_1_bil_B.jpg", res_B)


        print(" SONO IN 4.2.G! ")
        print("\n")
        for i in range(14):
            res_G= bf.bil_filter(s_G, 9,17,17)
        cv.imwrite(r"output/4_4_2_bil_G.jpg", res_G)


        print(" SONO IN 4.2.R! ")
        print("\n")
        for i in range(14):
            res_R= bf.bil_filter(s_R, 9,17,17)
        cv.imwrite(r"output/4_4_3_bil_R.jpg", res_R)

        ## 4.3- Unisco i canali per avere l'immagine a colori e filtrata
        print(" SONO IN 4.3! ")
        print("\n")
        ret= cv.merge((res_B,res_G,res_R))
        cv.imwrite(r"output/4_5_bil_img_short.jpg", ret)

        ## 4.4- applico bilinear_resize per riportare
        ## l' immagine alle dimensioni originali
        print(" SONO IN 4.4! BILINEAR RESIZE! ")
        print("\n")
        final= imf.bilinear_resize(ret, img.shape[0], img.shape[1])
        cv.imwrite(r"output/4_6_BIL_RES_IMAGE.jpg", final)

        ## 4.5- applico un filtro mediano 7x7
        ## per sfocare alcuni artefatti che si 
        ## possono essere verificati durante 
        ## il passo 4.4 (bilinear_resize)
        ## risultato quasi impercettibile
        print(" SONO IN 4.5! MEDIAN FILTER! ")
        print("\n")
        for i in range(3):
            final[:,:,i]= sb.median_filter(final[:,:,i], 7)
        cv.imwrite(r"output/4_7_BIL_IMAGE_MEDIAN_FILTER.jpg", final)


        ## 5) QUANTIZE COLOR
        ## cambiare il valore di ogni pixel
        ## con la seguente formula: pixel_nuovo= floor(pixel_corrente/a) * a
        ## per ottenere un effetto "cartoon"
        ## per un risultato migliore a=24
        print(" SONO IN 5! QUANTIZE COLOR! ")
        print("\n")
        color= imf.quantize_color(final, 24)
        cv.imwrite(r"output/5_quantize_color.jpg", color)


        ## 6) RECOMBINE
        ## sovrapporre all'immagine a colori i bordi
        ## dell' immagine "dilation" per ottenere
        ## il risultato finale
        print(" SONO IN 6! RECOMBINE! ")
        print("\n")
        result= imf.recombine(color,dilation_img)
        

        
        cv.imwrite(r"output/6_result.jpg", result)
        print("END! ")


        final= cv.imread("output/6_result.jpg")
        cv.imshow("Final result", final)
        cv.waitKey(0)
        


        cv.destroyAllWindows()
        quit()
    
    else:
        
        tkinter.messagebox.showinfo("ERRORE!","RIEMPI TUTTI I CAMPI!")
        cv.destroyAllWindows()







###############
### TKINTER ###
###############






window= tk.Tk()
window.geometry("600x700")
window.title("CARTOONIFYING_A_PHOTO")
window.resizable(False,False)
window.configure(background="green")
window.grid_columnconfigure(0,weight=1)


OptionList = [
"human2.jpg",
"tiger.jpg",
"bambino.jpg",
"dog.jpg",
"emma_stone.jpg",
"human.jpg",
"car.jpg"
]
variable = tk.StringVar(window)
variable.set(OptionList[0])


label0= tk.Label(window, text="BENVENUTO NEL MIO PROGETTO! CARTOONIFYING A PHOTO", font=("Helvetica", 25), background="green", fg="yellow")
label0.grid(row=0, column=0, sticky="N", padx=20, pady=70)

label1= tk.Label(window, text="Seleziona l'immagine su cui provare l'effetto ", font=("Helvetica", 15), background="green")
label1.grid(row=1, column=0, sticky="N", padx=20)
label2= tk.Label(window, text="es: tiger.jpg ",  font=("Helvetica", 10), background="green")
label2.grid(row=2, column=0, sticky="N", padx=20)
opt = tk.OptionMenu(window, variable, *OptionList)
opt.grid(row=3, column=0, sticky="WE", padx=50, pady=10)


label3= tk.Label(window, text="Seleziona il valore di sigma per il filtro gaussiano", font=("Helvetica", 15), background="green")
label3.grid(row=4, column=0, sticky="N", padx=20)
label4= tk.Label(window, text="consigliato: 3", font=("Helvetica", 10), background="green")
label4.grid(row=5, column=0, sticky="N", padx=20)
text_input2= tk.Entry()
text_input2.grid(row=6, column=0, sticky="WE", padx=50,pady=10)


label5= tk.Label(window, text="Seleziona il valore di 'lowThresholdRatio' ", font=("Helvetica", 15), background="green")
label5.grid(row=7, column=0, sticky="N", padx=20)
label6= tk.Label(window, text="consigliato: 0.01", font=("Helvetica", 10), background="green")
label6.grid(row=8, column=0, sticky="N", padx=20)
text_input3= tk.Entry()
text_input3.grid(row=9, column=0, sticky="WE", padx=50,pady=10)


label7= tk.Label(window, text="Seleziona il valore di 'HighThresholdRatio' ", font=("Helvetica", 15),background="green")
label7.grid(row=10, column=0, sticky="N", padx=20)
label8= tk.Label(window, text="consigliato: 0.1", font=("Helvetica", 10), background="green")
label8.grid(row=11, column=0, sticky="N", padx=20)
text_input4= tk.Entry()
text_input4.grid(row=12, column=0, sticky="WE", padx=50,pady=10)

button1= tk.Button(window, text="START",font=("Helvetica", 15) , command= esegui_codice, bd=4, activebackground="yellow" )
button1.grid(row=13, column=0, sticky="WE", padx=200, pady=20)




window.mainloop()


























