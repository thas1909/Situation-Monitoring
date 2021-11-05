from py2neo import Graph,Node,Relationship

from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout

import sys,os

objects= sys.argv[1].split(',')


# Pre-defined relations
relations = ["is-near","is-over","Has-a","Leads-to"]

labels = ["Object 1","Relation","Object 2"]
selected_label = {"obj_1":"a","rel":"b","obj_2":"c"}
class show_graph(App):

    def build(self):
        main_layout = BoxLayout(orientation='vertical')
        
        info = Label(   text="Create New Graphs by selecting options from the Pull Down Menu !",
                        color="black",
                        font_size='20sp',
                        size_hint=(1.0,0.3)                                            
                        )
        main_layout.add_widget(info)

        
        box = BoxLayout(orientation='horizontal',spacing=10,size_hint=(1.0,0.3)) 
        obj_1 = Button(text="Select Object_1")
        rel = Button(text="Select Relation")
        obj_2 = Button(text="Select Object_2")

        """ 
        obj_1 = Button(text="Select Object_1")
        rel = Button(text="Select Relation")
        obj_2 = Button(text="Select Object_2")

        dropdown_obj = self.get_drop_down(objects)
        dropdown_rel = self.get_drop_down(relations)
        
        obj_1.bind(on_release=dropdown_obj.open)
        obj_2.bind(on_release=dropdown_obj.open)
        rel.bind(on_release=dropdown_rel.open)
        
        dropdown_obj.bind(on_select=lambda instance, x: setattr(obj_1, 'text', x))
        dropdown_obj.bind(on_select=lambda instance, x: setattr(obj_1, 'text', x))
        dropdown_rel.bind(on_select=lambda instance, x: setattr(rel, 'text', x))


        box.add_widget(obj_1)
        box.add_widget(rel)
        box.add_widget(obj_2)
        """
        
        drop_obj1 = self.get_drop_down(objects)
        drop_obj2 = self.get_drop_down(objects)
        drop_rel = self.get_drop_down(relations)

        
        # show the dropdown menu when the main button is released
        # note: all the bind() calls pass the instance of the caller (here, the
        # mainbutton instance) as the first argument of the callback (here,
        # dropdown.open.).
        #mainbutton.bind(on_release=dropdown.open)
        obj_1.bind(on_release=drop_obj1.open)
        rel.bind(on_release=drop_rel.open)
        obj_2.bind(on_release=drop_obj2.open)

        # one last thing, listen for the selection in the dropdown list and
        # assign the data to the button text.
        drop_obj1.bind(on_select=lambda instance, x: self.get_clicked(obj_1,x,"obj_1"))
        drop_rel.bind(on_select=lambda instance, x: self.get_clicked(rel,x,"rel"))
        drop_obj2.bind(on_select=lambda instance, x: self.get_clicked(obj_2,x,"obj_2"))
       
        box.add_widget(obj_1)
        box.add_widget(rel)
        box.add_widget(obj_2)
        
        main_layout.add_widget(box)
        
        layout = AnchorLayout(anchor_x='center', anchor_y='center')
        btn = Button(text='Save',size_hint=(0.3,0.1))
        btn.bind(on_press = lambda alpha: self.create_graph(btn))
        layout.add_widget(btn)
        
        main_layout.add_widget(layout)

        #box1 = BoxLayout(orientation='horizontal',spacing=10)
        #main_layout.add_widget(box1)
                   
        return main_layout
    
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

    def get_clicked(self,button,value,selected_but_name):
        print(value)
        setattr(button, 'text', value)
        selected_label[selected_but_name]=value
        print(selected_label)

    def create_graph(self,but):
        # Access Graph
        g = Graph("bolt://neo4j:password@localhost:7687")
        a = Node("Object", name=selected_label["obj_1"]) # , age=33
        b = Node("Object", name=selected_label["obj_2"]) # , age=44
        relation = Relationship.type(selected_label["rel"])
        g.merge(relation(a, b), "Object", "name")# Primary label, Primary Key
        print("Graph created!!")
        setattr(but, 'text', "Graph Succesfully Created!!")


if __name__ == "__main__":
    show_graph().run()
