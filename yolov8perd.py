from ultralytics import YOLO
import json
d={"totalspace":20,"space":20,"remain":bool(True)}
model = YOLO("./runs/detect/train3/weights/best.pt")

results = model.predict(source="./9.jpg",save=True,conf=0.5)
names = model.names
car_id = list(names)[list(names.values()).index('car')]
detn=results[0].boxes.cls.tolist().count(car_id)
d["space"]=d["totalspace"]-detn
if d["space"]<=0:
    d["space"]=0
    d["remain"]=bool(False)
jdat=open("parking.json","w")
jdat.write(json.dumps(d))
jdat.close()