import imageio
import tifffile
import numpy as np
seed=42
np.random.seed(seed)
def convertToText(start, end):
    total = 0
    for img in  range(start,end,1):
        filename = 'images\geojson_' + str(img) + '.tif'

        try:
            image = tifffile.imread(filename)
            d = image.shape
            XTrain = image.reshape(d[0] * d[1], d[2])  # rehape row by row
            XTrain = np.delete(XTrain, 11, 1)
            XTrain = np.delete(XTrain, 10, 1)
            columns=XTrain.shape[1]
            outfilename = 'txtimages\geojson_' + str(img) + '.txt'
            total=total+XTrain.shape[0]
            with open(outfilename, "w") as f:
                print('rows:'+str(d[0]), file=f)
                print('columns:'+str(d[1]), file=f)
                index=0
                for i in range(d[0]):
                    for j in range(d[1]):
                        line=str(i+1)+","+str(j+1)
                        for k in range(columns):
                            line=line+','+str(XTrain[index,k])
                        index=index+1
                        line=line+",1"
                        print(line, file=f)
        except:
            print("missing "+filename)
    print("Read "+str(total)+" pixels")
def gobat(start,end):
    outfilename = 'txtimages\socrateR2.bat'
    line="java -XX:-UseGCOverheadLimit -jar socrate.jar -C \"scale,go,scale\" -D \"geojson_"
    #java -XX:-UseGCOverheadLimit -jar socrate.jar -C "scale,go,scale" -D "geojson_0" -P 50 -R 2 -E 0.03 -N 2707 -L "SVM"

    with open(outfilename, "w") as f:

        for img in  range(start,end,1):

            filename = 'images\geojson_' + str(img) + '.tif'

            try:
                image = tifffile.imread(filename)
                runbat = line + str(img) + "\"" + " -P 50 -R 2 -E 0.03 -N 2707 -L \"SVM\""
                print(runbat, file=f)
            except:
                print("missing " + filename)
        print("pause", file=f)

def imread(fileName):
    #fileName= "txtimages\geojson_"+str(index)+"ScaleGOS2Scale.txt"
    index=0

    with open(fileName) as f:
        for line in f.readlines():
            line = line.strip()
            if index==0:
                line=line.replace("rows:","")
                rows=int(line)
            if index==1:
                line = line.replace("columns:", "")
                columns = int(line)
            if(index==2):
                words=line.split(",")
                image = np.zeros((rows, columns, len(words)-3))#rimuovo x,y e classe finale
                i=int(words[0])-1
                j=int(words[1])-1
                for k in range(2,len(words)-1):
                    image[i,j,k-2]=float(words[k])
            if(index>2):
                words = line.split(",")

                i = int(words[0]) - 1
                j = int(words[1]) - 1
                for k in range(2, len(words) - 1):
                    image[i, j, k - 2] = float(words[k]);
            index=index+1
    #print(image.shape)
    return image

def convertToMask():
    start = 0
    end = 94
    fold = np.arange(start,end, 1)
    for i in fold:
        filename = 'images/geojson_' + str(i) + '.tif'
        try:
            image = tifffile.imread(filename)

            d = image.shape
            X = image.reshape(d[0] * d[1], d[2])  # rehape row by row
            Y=X[:,13]
            Y=np.nan_to_num(Y)
            Y=np.reshape(Y,(d[0],d[1]))
            imageio.imwrite("masks/mask_"+str(i)+".jpeg", (Y*255).astype(np.uint8))
        except        OSError as err:
            print("OS error:", err)


def main():
    start = 0
    end = 94
    #convertToText(start,end) #convert tiff in txt for socrate
    #gobat(start,end) #generate bat file to run socrate
    convertToMask()

if __name__ == "__main__":
    main()