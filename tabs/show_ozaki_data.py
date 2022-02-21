import sys,os
from kivy.config import Config
# Setting the initial window size of kivy. It can be resized during the process too. 
Config.set('graphics', 'width', '1440') 
Config.set('graphics', 'height', '720')
from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import *


#実際のアプリを動かす作業
data = sys.argv[1].split(",")
font = "../font/rounded-mgenplus-1cp-medium"

class show(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        # (Ww,Wh)=Window.size
        # h = Wh/len(data)
        for ind,datum in enumerate(data):

            warn = Label(   text= datum,
                            font_name=font,
                            color="black",
                            font_size='20sp',
                            size_hint=(1.0,0.1),                            
                            )
            
            layout.add_widget(warn)
            #layout.add_widget(Line(points=(100, 100, 200, 100),width=5)) # 0,h * (ind+1) , Ww, h * (ind+1)
            #layout.add_widget(Color(1, 0, 0, .5, mode='rgba'))
            
        return layout

if __name__ == "__main__":
    show().run()

