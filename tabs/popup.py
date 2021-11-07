from kivy.config import Config
# Setting the initial window size of kivy. It can be resizzed during the process too. 
Config.set('graphics', 'width', '400') 
Config.set('graphics', 'height', '250')
from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button


class popup(App):

    def build(self):
        main_layout = BoxLayout(orientation='vertical',spacing=20)
        
        warn = Label(   text="[b]Warning[/b]:[color=ff3333] Take Action Immediately!![/color]",
                        font_name='rounded-mgenplus-1cp-medium',
                        color="black",
                        font_size='20sp',
                        size_hint=(1.0,0.1),
                        markup = True,
                        )
        main_layout.add_widget(warn)

        details = Label(text=" [b]Danger Level[/b]: [color=ff3333]HIGH[/color]\n Window [color=DE9A1C]Is-Above[/color] Sofa [color=DE9A1C]LEADS-TO[/color] Fall\n Age: [b]Under 3 [/b] years \n No. Of. Related Accidents in the past: [b]100[/b] (Rank:[b]1[/b])",                        
                        font_name='rounded-mgenplus-1cp-medium',
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
        info.bind(on_press= lambda h:self.show_info("sofa","fall"))

        main_layout.add_widget(info)

        return main_layout

    def show_info(self,obj,risk):
        print("Parameter passd to show\/info function:",obj)
        from subprocess import Popen, PIPE
        process = Popen(['python', 'tabs/show_info.py',obj,risk], stdout=PIPE, stderr=PIPE)

if __name__ == "__main__":
    popup().run()   