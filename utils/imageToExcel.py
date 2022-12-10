
# import the modules
import os
from os import listdir
import xlwt
from xlwt import Workbook
 # Workbook is created
wb = Workbook()
character = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P", "Q", "R","S","T","U","V","W","X","Y","Z",
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
]  
# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')
sheet1.write(0, 0, 'image')
sheet1.write(0, 1, 'label')
i = 1
# get the path/directory
folder_dir = "./dataSet/ImgClustered2"
for images in os.listdir(folder_dir):
    #if images.startswith(character[k]):
    sheet1.write(i, 0, images+"")
    sheet1.write(i, 1, images[0]+'')
    i = i + 1
wb.save('./dataSet/modified1.xls')