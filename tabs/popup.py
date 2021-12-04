from kivy.config import Config
# Setting the initial window size of kivy. It can be resizzed during the process too. 
Config.set('graphics', 'width', '550') 
Config.set('graphics', 'height', '300')
from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

import sys
obj= sys.argv[1]
dist= sys.argv[2]
risk= sys.argv[3]
run = sys.argv[4]

font = "../font/rounded-mgenplus-1cp-medium"

class popup(App):

    def build(self):
        main_layout = BoxLayout(orientation='vertical',spacing=20)
        
        warn = Label(   text="[b]Warning[/b]:[color=ff3333] Take Action Immediately!![/color]",
                        font_name=font,
                        color="black",
                        font_size='20sp',
                        size_hint=(1.0,0.1),
                        markup = True,
                        )
        main_layout.add_widget(warn)

        details = Label(text=" [b]Danger Level[/b]: [color=ff3333]HIGH[/color]\n Child [color=DE9A1C]Is-Near[/color] [b]"+obj+"[/b] that [color=DE9A1C]Leads-to[/color] [b][color=ff3333]"+risk+"[/color][/b]\n Child is [color=ff3333]"+dist+"cms[/color] away from [b]"+obj+"[/b]!!\n",
        #Age: [b]Under 3 [/b] years \n No. Of. Related Accidents in the past: [b]100[/b] (Rank:[b]1[/b])",                        
                        font_name=font,
                        color="black",
                        font_size='16sp',
                        size_hint=(1.0,0.1),
                        markup = True,
                        )
        main_layout.add_widget(details)

        info = Button(text="More Info",
                      size_hint = (0.3,0.1),
                      pos_hint = {'center_x':0.5}
                      )
        info.bind(on_press= lambda h:self.show_info(obj,risk))

        main_layout.add_widget(info)

        return main_layout

    def show_info(self,obj,risk):
        print("Parameter passd to show\/info function:",obj)  
        dir ="samples/"
        from subprocess import Popen, PIPE
        process = Popen(["python show_info.py",obj,risk,dir], stdout=PIPE, stderr=PIPE)

if __name__ == "__main__":
    popup().run()   