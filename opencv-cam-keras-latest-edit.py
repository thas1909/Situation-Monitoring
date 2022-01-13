  # -*- coding: utf-8 -*-
# --- Kivy Imports ---
#:include cvcamera.kv
from kivy.config import Config
# Setting the initial window size of kivy. It can be resized during the process too. 
Config.set('graphics', 'width', '1280') 
Config.set('graphics', 'height', '720')
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.image import Image as Im
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.actionbar import ActionBar,ActionButton,ActionView,ActionPrevious
from kivy.uix.dropdown import DropDown

# --- Neo4j Setup ---
from py2neo import Graph,Node,Relationship
# Access Graph
#g = Graph("bolt://neo4j:password@localhost:7687")

# --- Tensorflow/Keras Imports ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from timeit import default_timer as timer

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
gpu_num=1

# --- Image Preocessing Library Imports ---
import cv2
from PIL import Image, ImageFont, ImageDraw


# --- Other Imports ---
import math
import numpy as np
import pyautogui # For screeen capture
import colorsys # For box coloring
from tabs.risk_data import risk_data 
from random import randint


# --- Initializing the variables ---
scale = 1 #To avoid out of frame issues etc. scale goes from 1 to 0.01 (100% to 1%)
risk_objects = [] # List to store the predicted objects that can pose risks
popup_status = 0 # To show alert when child approaches closer to dangerous objects

model_path = 'model_data/yolov3.h5'#trained_weights_final_blind.h5' # model path or trained weights path
anchors_path = 'model_data/yolo_anchors.txt'
classes_path = 'model_data/coco_classes.txt'#blind-class.txt'
score = 0.3
iou = 0.45
model_image_size = (416, 416) # fixed size or (None, None), hw

# Get Class Names
classes_path = os.path.expanduser(classes_path)
with open(classes_path) as f:
    class_names = f.readlines()
class_names = tuple(c.strip() for c in class_names)

# Get Anchors
anchors_path = os.path.expanduser(anchors_path)
with open(anchors_path) as f:
    anchors = f.readline()
anchors = [float(x) for x in anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)

# Generate ()
model_path = os.path.expanduser(model_path)
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

# Load model, or construct model and load weights.
num_anchors = len(anchors)
num_classes = len(class_names)
is_tiny_version = num_anchors==6 # default setting
try:
    yolo_model = load_model(model_path, compile=False)
except:
    yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
        if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
    yolo_model.load_weights(model_path) # make sure model, anchors and classes match
else:
    assert yolo_model.layers[-1].output_shape[-1] == \
        num_anchors/len(yolo_model.output) * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes'
#print('{} model, anchors, and classes loaded.'.format(model_path))

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
            for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list( map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors) 
            )
np.random.seed(10101)  # Fixed seed for consistent colors across runs.
np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
np.random.seed(None)  # Reset seed to default.

# Generate output tensor targets for filtered bounding boxes.
input_image_shape = K.placeholder(shape=(2, ))

# if gpu_num>=2:
#     yolo_model = multi_gpu_model(yolo_model, gpus=gpu_num)

boxes, scores, classes = yolo_eval( yolo_model.output, anchors,
                                    len(class_names), input_image_shape,
                                    score_threshold = score, iou_threshold = iou )

sess = K.get_session() # Starting a new keras session...

ppp=1

# .... To be Developed ....

# def get_data_from_neo4j():
#     print("Objects:")
#     dangerousObjects = g.run("""MATCH (n) WHERE EXISTS(n.name) RETURN DISTINCT "node" as entity, n.name AS name LIMIT 25 UNION ALL MATCH ()-[r]-() WHERE EXISTS(r.name) RETURN DISTINCT "relationship" AS entity, r.name AS name""")
#     for record in dangerousObjects:
#         print(record["name"])

#     print("\nRisks:")
#     risks = g.run("""MATCH (n) WHERE EXISTS(n.type) RETURN DISTINCT "node" as entity, n.type AS type LIMIT 25 UNION ALL MATCH ()-[r]-() WHERE EXISTS(r.type) RETURN DISTINCT "relationship" AS entity, r.type AS type""")
#     for record in risks:
#         print(record["type"])

main_widget_count = 0

class CvCamera(App):
    
    def build(self): #UIの構築等
        global main_widget_count
        # ButtonやSlider等は基本size_hintでサイズ比率を指定(絶対値の時はNoneでsize=)
        # Verticalの中に置くhorizontalなBoxLayout (ここだけ2column)

        # For Horizontal Zoom
        """
        layout2 = BoxLayout(orientation='horizontal', size_hint=(1.0, 0.1))
        self.s1Label = Label(text = 'Zoom', size_hint=(0.3, 1.0), halign='center',color="blue",font_size='30sp',markup=True)
        slider1 = Slider(size_hint=(0.7, 1.0),min=1, max=99,step=1)
        slider1.bind(value=self.slideCallback)

        #layout3 = BoxLayout(orientation='vertical', pos_hint={'x':0.8,'y': 0}) #size_hint=(0.1, 1.0)
        
        graph_button = Button(background_normal = 'graph-button.png',
                    size_hint = (0.08,0.12), 
                    pos_hint = {'right': 1, 'top': 1},
                    background_color = (1,1,1,1) )
        graph_button.bind(on_press= lambda h:self.show_graph())        
        """
        """
        # For Vertical Zoom 
        slider2 = Slider(orientation="vertical",size_hint=(0.7, 1.0),min=1, max=99,step=1) 
        slider2.bind(value=self.slideCallback)
        self.s2Label = Label(text = 'Slider2', halign='center',color='blue') 
        # # 日本語フォントを使いたいときはfont_nameでフォントへのパス
        #zoomIN = Button(text='Zoom IN', color = "red",pos_hint={'x':.8,'y':.1},size_hint=(0.2, 0.1), font_name='rounded-mgenplus-1cp-medium')
        #zoomOUT = Button(text='Zoom OUT', color = "blue",pos_hint={'x':.8,'y':.2},size_hint=(0.2, 0.1), font_name='rounded-mgenplus-1cp-medium')

        #zoomIN.bind(on_release = lambda x:self.buttonCallback(zoomIN.text)) #bindでイベントごとにコールバック指定
        #zoomOUT.bind(on_release = lambda x:self.buttonCallback(zoomOUT.text)) #bindでイベントごとにコールバック指定
        """
        # Imageに後で画像を描く
        self.img1 = Im(allow_stretch = True,keep_ratio = False)
        
        # Layoutを作ってadd_widgetで順次モノを置いていく(並びは置いた順)
        self.layout = FloatLayout()
        self.layout.add_widget(self.img1)
       
        # self.layout.add_widget(layout2)
        # self.layout.add_widget(graph_button)
        # layout2.add_widget(self.s1Label)
        # layout2.add_widget(slider1)

        # self.layout.add_widget(layout3)
        # layout3.add_widget(self.s2Label)
        # layout3.add_widget(slider2)
        # 1columnに戻る
        #self.layout.add_widget(zoomIN)
        #self.layout.add_widget(zoomOUT)
        # act_view = ActionView(use_separator=True)
        # act_bar = ActionBar(pos_hint = {'top':1})
        # act_btn = ActionButton(text="sofa",important=True)
        # act_prev = ActionPrevious(title="Action Bar",with_previous=False)
        # act_view.add_widget(act_prev)
        # act_view.add_widget(act_btn)
        # act_bar.add_widget(act_view)
        # self.layout.add_widget(act_bar)

        # DropDown of general dangerous objects
        btn_layout = BoxLayout(orientation='vertical',size_hint=(0.15,0.1),pos_hint = {'right': 1, 'top': .2})
        general_objects = ["tvmonitor","sofa","bed","cup","window","blind"]
        obj_btn = Button(text="Objects")
        drop_obj = self.get_drop_down(general_objects)
        obj_btn.bind(on_release=drop_obj.open)
        drop_obj.bind(on_select=lambda instance, x: self.show_info(x,risk_data[x][0]))
        btn_layout.add_widget(obj_btn)
        self.layout.add_widget(btn_layout)
        main_widget_count = len(self.layout.children)
        print("////////////////// Len of widget:",main_widget_count)

        # 更新スケジュールとコールバックの指定
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.layout

    def get_drop_down(self,objects):
            # create a dropdown with 10 buttons
            drop_obj1 = DropDown()

            for index in objects:
                # When adding widgets, we need to specify the height manually
                # (disabling the size_hint_y) so the dropdown can calculate
                # the area it needs.

                btn = Button(text=index, size_hint_y=None, height=44)

                # for each button, attach a callback that will call the select() method
                # on the dropdown. We'll pass the text of the button as the data of the
                # selection.
                btn.bind(on_release=lambda btn: drop_obj1.select(btn.text))

                # then add the button inside the dropdown
                drop_obj1.add_widget(btn)  
            return drop_obj1

    def buttonCallback(self, value):
        # 何かのフラグに使える
        global scale
        print('Buttn <%s> is pressed.' % (value))

        if value == "Zoom IN":
            scale+=2
            if scale>50:
                scale=50 

        if value == "Zoom OUT":
            scale-=2
            if scale<0:
                scale=0
        print(scale)

    def slideCallback(self, instance, value):
        # Slider横のLabelをSliderの値に
        self.s1Label.text = '[b]Zoom %s[/b]' % int(value)
        global scale
        
        scale=round((1.0 - int(value)/100.0),2)
        
        print(scale) 

    def create_popup(self,obj,dist,risk):
        # global popup_status
        # popup_status=1
        # popup = Popup(title='Warning:  Take Action Immediately ',
        #         content=Label(text='[b]Danger Level[\b]:\n  [color=ff3333]Window Is-Above Sofa LEADS-TO Fall[/color] \n Age: Under 3 years \n No. Of. Related Accidents in the past: 100 (Rank:1)'),
        #         size_hint=(None, None), size=(400, 400))
        # popup.open()
        global ppp
        if ppp == 1:
            run = "python tabs/show_info.py "+obj+" "+risk+" "+"samples/"
            from subprocess import Popen, PIPE
            process = Popen(['python', 'tabs/popup.py',obj,dist,risk,run], stdout=PIPE, stderr=PIPE)
            ppp=2

    def createButton(self,x,y,obj,risk):
        
        # Get the current kivy window height & width 
        (Ww,Wh)=Window.size
        Rx = float(Ww/self.width)
        Ry = float(Wh/self.height)
        dir="samples/"
       
        # Info-buttons with bounding box version 
        info_buttons = ["i-orange.png","i-yellow.png"]
        but_path = 'icons/edit/' + info_buttons[0]
        bt = Button(background_normal = but_path,
                    size_hint = (0.05,0.075), #0.05,0.075
                    pos = (x*Rx-20, (720 - (y+32))*Ry) # For Kinect : (720 - (y+55))*Ry)  # For Screen-Record: (720 - (y+32))*Ry)
                    )
        # Only Rabbit version
        #rabbit_buttons = ("up","down","left","right")
        #but_path = 'icons/rabbit/' + rabbit_buttons[randint(0,len(rabbit_buttons)-1)] + '.png'
        #but_path = 'icons/rabbit/down.png'

        # bt = Button(background_normal = but_path,
        #             size_hint = (0.08,0.13), #0.05,0.075
        #             pos = (x*Rx-20, (720 - (y))*Ry) # For Kinect : (720 - (y+55))*Ry)  # For Screen-Record: (720 - (y+32))*Ry)
        #             )
        bt.bind(on_press= lambda h:self.show_info(obj,risk,dir))
        self.layout.add_widget(bt)
        
    def show_info(self,obj,risk,dir):
        print("Parameter passd to show info function:",obj)
        from subprocess import Popen, PIPE
        process = Popen(['python', 'tabs/show_info.py',obj,risk,dir], stdout=PIPE, stderr=PIPE)
        
    def show_graph(self):        
        from subprocess import Popen, PIPE
        values = ','.join(risk_objects)
        print("Risk objects",values)
        process = Popen(['python', 'tabs/graph.py',values], stdout=PIPE, stderr=PIPE)

    def show_logo(self,x,y):
        (Ww,Wh)=Window.size
        Rx = float(Ww/self.width)
        Ry = float(Wh/self.height)
        bt = Button(background_normal = 'icons/original/imresizer.png',
                    size_hint = (0.05,0.1), #0.05,0.075
                    pos = (Rx*x-60,Wh-Ry*(y+45))
                    )
        self.layout.add_widget(bt)        

    def object_buttons(self):
        general_objects = ["tvmonitor","sofa","bed","cup","window","blind"]
        act_view = ActionView(use_separator=True)
        act_bar = ActionBar(pos_hint = {'top':1})
        act_btn = ActionButton(text="sofa",important=True)
        act_view.add_widget(act_btn)
        act_bar.add_widget(act_view)

    def update(self,dt):
        # 基本的にここでOpenCV周りの処理を行なってtextureを更新する
        global scale,popup_status,main_widget_count

        idx=0
        #print("**********************",len(self.layout.children),main_widget_count)
        for child in self.layout.children[:-main_widget_count]:
            self.layout.remove_widget(child)                
            #print("child {}: {}".format(idx,child))            
            idx=idx+1
        
        
        # --- To detect objects via real-time skype or zoom video call ---
        # Skype : NO Beta screem (old one) & select "Show Only participants with video screen"
        im1 = pyautogui.screenshot(region=(0,140, 1920, 830)) #[1920,1080]
        #Google-Meet
        ##im1 = pyautogui.screenshot(region=(0,215, 1920, 755))
        frame = np.array(im1)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        (H, W) = frame.shape[:2]
        #print(pyautogui.position())
               
        self.height = H
        self.width = W

        #prepare the crop
        centerX,centerY=int(W/2),int(H/2)
    
        radiusX,radiusY= int(scale*centerX),int(scale*centerY)

        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY
        
        #cv2.imshow("original",frame)
        cropped = frame[minY:maxY , minX:maxX]
        #cv2.imshow("cropped",cropped)
        # The modified frame
        frame = cv2.resize(cropped, (W,H),interpolation = cv2.INTER_AREA)
                   
        frame = frame[:,:,(2,1,0)]
        image = Image.fromarray(frame)

        # if self.model_image_size != (None, None):
        #     assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        #     boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # else:
        new_image_size = (image.width - (image.width % 32),
                            image.height - (image.height % 32))
        
        boxed_image = letterbox_image(image, new_image_size)
        
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes_x, out_scores_x, out_classes_x = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # Extract only unique labels 
        out_boxes =[]
        out_classes=[]
        #out_scores=[]
        for i,c in enumerate(out_classes_x):
            if c not in out_classes:
                out_classes.append(c)
                out_boxes.append(out_boxes_x[i])
                #out_scores.append(out_scores_x[i])

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #print("outclasses=",out_classes,type(out_classes),out_boxes,out_scores)

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                    
        thickness = (image.size[0] + image.size[1]) // 300

        print(out_classes)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
                     
                                     
            for k,v in risk_data.items():
                
                if predicted_class in k:
                    
                    #risk="Risk: {}".format(v)
                    draw = ImageDraw.Draw(image)
                    #print("Value of V:",v[0],v[1])

                    #label = "{}: {:.2f}  {}".format(predicted_class,score,risk)
                    #label = '{} {:.2f}'.format(predicted_class, score)
                    #cv2.putText(frame, text, (left, top - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                
                    #label_size = draw.textsize(predicted_class, font)
                    print(predicted_class, (left, top), (right, bottom))

                    # if top - label_size[1] >= 0:
                    #     text_origin = np.array([left, top - label_size[1]])
                    # else:
                    #     text_origin = np.array([left, top - 1])
                    
                    # l = 10 # Window size where depth values will be considered
                    
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=colors[c]
                        )

                        # # box showing the region where depth values will be considered
                        # draw.rectangle(
                        #     [mid_x-l + i, mid_y-l + i, mid_x+l - i, mid_y+l - i],
                        #     outline=colors[c]
                        # )
                        
                    r=20
                    for i in range(v[1]):
                        R = right-r*(i)
                        L = R-r
                        draw.ellipse((L,top-2*r,R,top-r), fill=(255, 128, 179))

                    #Additional logo insert
                   
                    if "fall" in v[0]:
                        self.show_logo(int(right),int(top)+30)
                    # draw.rectangle
                    #     [tuple(text_origin), tuple(text_origin + label_size)],
                    #     fill=colors[c])

                    #draw.text(text_origin, label, fill=colors[c], font=font)
                    del draw
                    
                    # To create i-buttons...
                    self.createButton(int(left)+thickness,int(top),predicted_class,v[0])    

                    # To send the objects detected to show it the drop-down list in Graph
                    if predicted_class not in risk_objects:
                        risk_objects.append(predicted_class) 
                    

        print("*******")            
        frame = np.array(image)[:,:,(2,1,0)]
        #cv2.imshow("YOLOv2", np.array(out_img))

        frame = cv2.flip(frame, 0)
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(frame.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture1

if __name__ == '__main__':    
    CvCamera().run()


""" 
## Neo4j examples
def Neo4j():
    from py2neo import Graph,Node,Relationship
    g = Graph("bolt://neo4j:password@localhost:7687")

    a = Node("Person", name="Alice", age=33)
    b = Node("Person", name="Bob", age=44)
    KNOWS = Relationship.type("KNOWS")
    #g.merge(KNOWS(a, b), "Person", "name")# Primary label, Primary Key

    # Pre-defined relations
    relations = ["is-near","is-over","Has-a","Leads-to"] 

## COMMENT A
def close_session(self):
    self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
    
def detect_img(yolo):
    while True:
        ret, image = cap.read()
        if cv2.waitKey(10) == 27:
            break
        h, w = image.shape[:2]
        rh = int(h * camera_scale)
        rw = int(w * camera_scale)
        image = cv2.resize(image, (rw, rh))
        image = image[:,:,(2,1,0)]
        image = Image.fromarray(image)
        r_image = yolo.detect_image(image)
        out_img = np.array(r_image)[:,:,(2,1,0)]
        cv2.imshow("YOLOv2", np.array(out_img))
    yolo.close_session()

def cv2_dnn_darknet():
    configPath="yolo-coco/yolov3.cfg"
    weightsPath="yolo-coco/yolov3.weights"
    conf=0.1
    thresh=0.1
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")


    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Using GPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print(cv2.cuda.getCudaEnabledDeviceCount())

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    #start = time.time()
    layerOutputs = net.forward(ln)
    #end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > conf:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf,thresh)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            for k,v in risk_data.items():
                if LABELS[classIDs[i]] in k:
                    obj=LABELS[classIDs[i]]
                    risk="Risk: {}".format(v)
                    self.createButton(x,720-y,obj,v)                       
                            
                else:
                    risk=""
                
                text = "{}: {:.4f}  {}".format(LABELS[classIDs[i]],
                    confidences[i],risk)
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
"""