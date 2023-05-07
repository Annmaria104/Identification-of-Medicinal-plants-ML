from tkinter import *
import time
import re
#Import scikit-learn metrics module for accuracy calculation
import pickle
from PIL import Image, ImageTk  
import cv2
from tkinter.filedialog import askopenfile
from Testing import test_leaf

def pp(a):
    global mylist
    mylist.insert(END, a)



def predict(val):
    global mylist,text_box
    print(val)
    text_box.delete('1.0', END)
    result,useval=test_leaf(val)
    uses="Medicinal Plant Uses\n================================\n"+useval
    
    root.after(500, lambda : pp("Uploading  image"))
    root.after(1700, lambda : pp("Loading Trained Model"))
    root.after(2000, lambda : pp("Image feature extraction"))
    root.after(2500, lambda : pp("Prediction using Loaded model"))
    root.after(2800, lambda : pp("Result : "+result))
    root.after(3000, lambda : pp("============================"))
    root.after(3100, lambda :shrslt.config(text=result,fg="red"))
    root.after(3100, lambda :shrslt.config(text=result,fg="red"))
    root.after(3100, lambda :text_box.insert('end', uses))
    
        
    
        
def browseim():
    global cimg,shrslt,E1
    path = askopenfile()
    n=path.name 
    print(n)
    E1.delete(0,"end")
    E1.insert(0, n)
    
def userHome():
    global root, mylist,shrslt,E1,text_box
    root = Tk()
    root.geometry("1200x700+0+0")
    root.title("Home Page")

    image = Image.open("bck2.jpg")
    image = image.resize((1200, 700), Image.ANTIALIAS) 
    pic = ImageTk.PhotoImage(image)
    lbl_reg=Label(root,image=pic,anchor=CENTER)
    lbl_reg.place(x=0,y=0)
  
    #-----------------INFO TOP------------
    lblinfo = Label(root, font=( 'aria' ,20, 'bold' ),text="Medicinal Plant Identification",fg="white",bg="#000955",bd=10,anchor='w')
    lblinfo.place(x=420,y=50)
 
    lblinfo3 = Label(root, font=( 'aria' ,20 ),text="input image path",fg="#000955",anchor='w')
    lblinfo3.place(x=180,y=200)
    E1 = Entry(root,width=30,font="veranda 20")
    E1.place(x=50,y=260)
    mylist = Listbox(root,width=60, height=15,bg="white")

    mylist.place( x = 700, y = 130 )

    text_box = Text(root,height=12,width=50)
    text_box.place( x = 700, y = 450 )


    btntrn=Button(root,padx=10,pady=2, bd=4 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Browse", bg="red",command=lambda:browseim())
    btntrn.place(x=180, y=300)
    btnhlp=Button(root,padx=80,pady=8, bd=6 ,fg="white",font=('ariel' ,10,'bold'),width=7, text="Test", bg="blue",command=lambda:predict(E1.get()))
    btnhlp.place(x=150, y=400)
    lblinfo1 = Label(root, font=( 'aria' ,20, ),text="Result :",fg="red",bg="white",anchor=W)
    lblinfo1.place(x=100,y=480)
    shrslt = Label(root, font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
    shrslt.place(x=200,y=480)

   

    def qexit():
        root.destroy()
     

    root.mainloop()


userHome()