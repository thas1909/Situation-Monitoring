from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.carousel import Carousel
from kivy.uix.image import AsyncImage

import sys,os
obj= sys.argv[1]
risk= sys.argv[2]
directory= sys.argv[3] #"samples/"


font = "../font/rounded-mgenplus-1cp-medium"
# kv = Builder.load_string("""
# <BoxLayout>:
#     canvas.before:
#             Color:
#                 rgba: 1, 1, 1, 1
#             Rectangle:
#                 pos: self.pos
#                 size: self.size
# """)

class show_info(App):

    def build(self):
        main_layout = BoxLayout(orientation='vertical')
        
        warn = Label(   text="[b]注意[/b]：「"+"[color=ff3333]"+obj.upper()+"[/color]"+"」に関して、以下に示すような危険性が予測されます！! ",
                        font_name=font,
                        color="black",
                        font_size='20sp',
                        size_hint=(1.0,0.1),
                        markup = True,
                        )
        main_layout.add_widget(warn)
        details = Label(text="[b]RISK[/b]: "+risk,                        
                        font_name=font,
                        color="red",
                        font_size='20sp',
                        size_hint=(1.0,0.1),
                        markup = True,
                        )
        main_layout.add_widget(details)

        #layout = BoxLayout(orientation='horizontal',spacing=10)
        # layout = GridLayout(cols=2,spacing=10)      

        

        # for filename in os.listdir(directory):
        #     if obj in filename: 
        #             fil = os.path.join(directory, filename)
        #             if fil is not None:
        #                 img=Image(source=fil,
        #                         allow_stretch=True,                             
        #                         ) 
        #                 layout.add_widget(img)
        #             else:
        #                 layout.add_widget(Label(text="情報がありません！",                        
        #                                         font_name='rounded-mgenplus-1cp-medium',
        #                                         color="red",
        #                                         font_size='20sp',
        #                                         size_hint=(1.0,0.4),
        #                                         ))
        # main_layout.add_widget(layout)        
        
        carousel = Carousel(direction='right')
        for filename in os.listdir(directory):
            if obj in filename: 
                fil = os.path.join(directory, filename)
                if fil is not None:
                    image = AsyncImage(source=fil, allow_stretch=True)
                    carousel.add_widget(image)
                    print(carousel.index)
        main_layout.add_widget(carousel)
        return main_layout


if __name__ == "__main__":
    show_info().run()
