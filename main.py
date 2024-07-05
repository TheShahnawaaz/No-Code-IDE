from prettytable import PrettyTable
from kivy.clock import Clock
import pickle
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty,
    StringProperty,
    BooleanProperty,
    ListProperty,
    ObjectProperty,
)
import os
# import fileinput
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.treeview import TreeView, TreeViewLabel
from kivy.core.window import Window
from kivy.uix.spinner import Spinner
import os
import ast
import importlib
import pandas as pd
import numpy as np
# import TabbledPanelItem
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.button import Button
import re
from kivy.uix.dropdown import DropDown
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
import inspect
from kivy.uix.actionbar import ActionBar
import json


def extract_parameters(doc_text):
    # Define regular expressions for identifying different sections and extracting parameters
    section_pattern = re.compile(r'^\s*([A-Za-z]+)\s*$', re.MULTILINE)
    param_pattern = re.compile(
        r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', re.MULTILINE)

    # Find all sections in the documentation
    sections = section_pattern.findall(doc_text)

    # Initialize a dictionary to store parameters for each section
    parameters_dict = {}

    # Iterate through sections and extract parameters
    for i in range(len(sections) - 1):
        section_start = doc_text.find(sections[i])
        section_end = doc_text.find(sections[i + 1])
        section_text = doc_text[section_start:section_end].strip()

        # Find all parameters in the current section
        parameters = param_pattern.findall(section_text)

        # Store parameters in the dictionary
        parameters_dict[sections[i]] = parameters

    # Extract parameters for the last section
    section_start = doc_text.find(sections[-1])
    section_text = doc_text[section_start:].strip()
    parameters = param_pattern.findall(section_text)
    parameters_dict[sections[-1]] = parameters

    # Extract parameters from the "Parameters" section
    parameters_list = parameters_dict.get('Parameters', [])

    return parameters_list


def extract_parameters_with_descriptions(doc_text):
    # Define regular expressions for identifying different sections and extracting parameters
    section_pattern = re.compile(r'^\s*([A-Za-z]+)\s*$', re.MULTILINE)
    param_pattern = re.compile(
        r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', re.MULTILINE)

    # Find all sections in the documentation
    sections = section_pattern.findall(doc_text)

    # Initialize a dictionary to store parameters and their descriptions for each section
    parameters_dict = {}

    # Iterate through sections and extract parameters
    for i in range(len(sections) - 1):
        section_start = doc_text.find(sections[i])
        section_end = doc_text.find(sections[i + 1])
        section_text = doc_text[section_start:section_end].strip()

        # Find all parameters in the current section
        parameters = param_pattern.findall(section_text)

        # Store parameters in the dictionary
        parameters_dict[sections[i]] = {
            param[0]: param[1] for param in parameters}

    # Extract parameters for the last section
    section_start = doc_text.find(sections[-1])
    section_text = doc_text[section_start:].strip()
    parameters = param_pattern.findall(section_text)
    parameters_dict[sections[-1]] = {param[0]: param[1] for param in parameters}

    # Extract parameters and their descriptions from the "Parameters" section
    parameters_with_descriptions = parameters_dict.get('Parameters', {})

    return parameters_with_descriptions


def extract_parameters_(doc_text):
    # Define regular expressions for identifying different sections and extracting parameters
    section_pattern = re.compile(r'^\s*([A-Za-z]+)\s*$', re.MULTILINE)
    # Define a regular expression pattern to extract parameter information
    param_pattern = re.compile(
        r'(?P<param>\w+)\s*:\s*(?P<type>[\w\s\[\],]+)\s*\n\s*(?P<desc>.*?)(?=\n\s*(?:\w+\s*:|$))', re.DOTALL)

    # Search for parameter information in the documentation
    # Find all sections in the documentation
    sections = section_pattern.findall(doc_text)

    # Initialize a dictionary to store parameters for each section
    parameters_dict = {}

    # Iterate through sections and extract parameters
    for i in range(len(sections) - 1):
        section_start = doc_text.find(sections[i])
        section_end = doc_text.find(sections[i + 1])
        section_text = doc_text[section_start:section_end].strip()+'\n'

        # Find all parameters in the current section
        # parameters = param_pattern.findall(section_text)
        param_matches = param_pattern.finditer(section_text)
        # Iterate over parameter matches and populate the dictionary
        parameters = {}
        for match in param_matches:
            param_name = match.group('param')
            param_type = match.group('type').strip()
            param_desc = match.group('desc').strip()

            parameters[param_name] = {'type': param_type, 'desc': param_desc}
        # Store parameters in the dictionary
        parameters_dict[sections[i]] = parameters

    # Extract parameters for the last section
    section_start = doc_text.find(sections[-1])
    section_text = doc_text[section_start:].strip()
    parameters = param_pattern.findall(section_text)
    # Find all parameters in the current section
    # parameters = param_pattern.findall(section_text)
    param_matches = param_pattern.finditer(section_text)
    # Iterate over parameter matches and populate the dictionary
    parameters = {}
    for match in param_matches:
        param_name = match.group('param')
        param_type = match.group('type').strip()
        param_desc = match.group('desc').strip()

        parameters[param_name] = {'type': param_type, 'desc': param_desc}
    # Store parameters in the dictionary
    parameters_dict[sections[-1]] = parameters

    # Extract parameters from the "Parameters" section
    parameters_list = parameters_dict.get('Parameters', [])

    return parameters_list


def execute_code_from_file(file_path):
    with open(file_path, 'r') as file:
        source_code = file.read()

    try:
        # Parse the abstract syntax tree (AST) of the code
        parsed_ast = ast.parse(source_code)

        # Extract import statements from the AST
        import_statements = [node for node in parsed_ast.body if isinstance(
            node, ast.Import) or isinstance(node, ast.ImportFrom)]

        # print("import_statements", import_statements)
        # Import the regular modules dynamically
        for import_stmt in import_statements:
            if isinstance(import_stmt, ast.Import):
                for alias in import_stmt.names:
                    module_name = alias.name
                    try:
                        imported_module = importlib.import_module(module_name)
                        globals()[
                            alias.asname if alias.asname else module_name] = imported_module
                        # print("IMPORT\n",module_name, imported_module, globals())
                    except ImportError:
                        print(f"Error importing module: {module_name}")

            elif isinstance(import_stmt, ast.ImportFrom):
                module_name = import_stmt.module
                for alias in import_stmt.names:
                    try:
                        full_module_name = f"{module_name}"

                        imported_module = importlib.import_module(
                            full_module_name)
                        globals()[alias.asname if alias.asname else alias.name] = getattr(
                            imported_module, alias.name)
                        # print("IMPORT FROM\n",full_module_name, imported_module, globals(),"\n\n")
                    except ImportError:
                        print(f"Error importing module: {full_module_name}")

        # Execute the code
        exec(source_code, globals(), locals())
    except Exception as e:
        print(f"Error executing code from file '{file_path}': {e}")


# Window.clearcolor = (1, 1, 1, 1)
Window.maximize()


def remove_extra_spaces(text):
    words = text.split()
    return ' '.join(words)


class Model_Tabbed_Panel_Item(TabbedPanelItem):

    input_tabbed_panel = ObjectProperty(None)

    def __init__(self, text, tab_data, main_widget, **kwargs):
        super(Model_Tabbed_Panel_Item, self).__init__(text=text, **kwargs)
        # self.text = tab_name
        print(tab_data)
        self.tab_data = tab_data
        self.main_widget = main_widget
        self.ids.model_name_label.text = tab_data[1]
        # self.ids.label_1.text = tab_data[0]
        # self.ids.label_2.text = tab_data[2].split('#')[1]
        # self.ids.label_3.text = tab_data[3]
        class_parts = tab_data[2].split('#')[1].split('.')
        module_name = '.'.join(class_parts[:-1])
        class_name = class_parts[-1]

        module = __import__(module_name, fromlist=[class_name], level=0)
        print("module", module)

        self.model_class = getattr(module, class_name)
        # self.model_instance = model_class()

        model_doc = inspect.getdoc(self.model_class)

        self.ids.label_4.text = str(model_doc) + "\n\nEnd\n--------"

        print("Model Doc", str(model_doc) + "\n\nEnd\n--------\n\n")
        print("Model.__doc__", self.model_class.__doc__)

        self.model_docs = extract_parameters_(model_doc)

        for param, info in self.model_docs.items():
            print(param, info)
            self.input_tabbed_panel.add_tab(param, info)


    def generate_input(self):
        for i in self.input_tabbed_panel.tab_list:
            print(i.text, i.val, i.value_to_display, i.isArray)
            print()
            print()
            print()

    def run_model(self):


        try :
            self.generate_input()
            print("Running Model")

            inputs = {}

            for i in self.input_tabbed_panel.tab_list:
                if i.val is not None:
                    # val is a list of size one list , then flatten it
                    # if i.val is a list
                    if isinstance(i.val, list):

                        if len(i.val) == 1:
                            # [{'file_path': 'F:\\IIM\\2024_01_20\\Try\\data_temp8.xlsx', 'sheet_name': 'Sheet1', 'column_name': 'y'}]
                            file_path = i.val[0]['file_path']
                            sheet_name = i.val[0]['sheet_name']
                            column_name = i.val[0]['column_name']

                            # Read the file and get the column data
                            if file_path.endswith(".xlsx"):
                                excel_data = pd.read_excel(
                                    file_path, sheet_name=sheet_name)
                                inputs[i.text] = excel_data[column_name].tolist()
                            elif file_path.endswith(".csv"):
                                excel_data = pd.read_csv(file_path)
                                inputs[i.text] = excel_data[column_name].tolist()

                        else:  # pass the transposed list

                            multi_col_data = []
                            for col in i.val:

                                if col['file_path'].endswith(".xlsx"):
                                    excel_data = pd.read_excel(
                                        col['file_path'], sheet_name=col['sheet_name'])
                                    multi_col_data.append(
                                        excel_data[col['column_name']].tolist())
                                elif col['file_path'].endswith(".csv"):
                                    excel_data = pd.read_csv(col['file_path'])
                                    multi_col_data.append(
                                        excel_data[col['column_name']].tolist())

                            inputs[i.text] = list(map(list, zip(*multi_col_data)))

                            # inputs[i.text] = list(map(list, zip(*i.val)))
                    else:
                        inputs[i.text] = i.val

            self.model_inputs = inputs
            print(inputs)

            print(self.model_class)
            self.model_instance = self.model_class(**inputs)

            print(self.model_instance)

            self.model_fitted = self.model_instance.fit()

            print(self.model_fitted.summary())

            self.ids.summary_label.text = str(self.model_fitted.summary())

            run_date = pd.to_datetime('today').strftime('%Y-%m-%d')

            # Save the prediction in the MODEL_RUNS folder
            run_save_path = os.path.join(
                self.main_widget.CURRENT_MODEL_RUNS_DIR, self.text + "_" + run_date + ".csv")
            print(run_save_path)

            # try:
            self.prediction = self.model_fitted.predict(inputs["exog"])
            self.save_prediction_in_excel_helper(run_save_path)
            print("Saved Prediction in", run_save_path)
            # except:
            #     print("Not able to predicct")

        except Exception as e:
            print("Error Running Model", e)
            MessageBox(title="Error", message="Error Running Model\n"+str(e), message_type="error").open()
            

    def save_prediction_in_excel(self):

        save_popup = FileSavePopup(model_item=self)
        save_popup.open()

    def save_prediction_in_excel_helper(self, fullpath):
        try:
            print("Saving to", fullpath)
            # Save the DataFrame with predicted values to Excel
            print(type(self.prediction))  # nd.array

            print(self.prediction)
            # Save this in the local file
            # self.prediction is a numpy array
            # convert it to a pandas dataframe
            df = pd.DataFrame(self.prediction, columns=["Predicted"])
            print(df)

            if fullpath.endswith(".xlsx"):
                df.to_excel(fullpath, index=False)
            elif fullpath.endswith(".csv"):
                df.to_csv(fullpath, index=False)

            print("Saved to", fullpath)
        
        except Exception as e:
            print("Error Saving Prediction", e)
            MessageBox(title="Error", message="Error Saving Prediction\n"+str(e), message_type="error").open()
            return


class FileSavePopup(Popup):

    # def on_dismiss(self, instance):
    #     # Handle dismissal (optional)
    #     pass
    def __init__(self, model_item, **kwargs):
        super().__init__(**kwargs)
        self.model_item = model_item
        # Fill the filename input with the model name and prediction and timestamp
        self.ids.filename_input.text = model_item.text + "_prediction_" + \
            pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M-%S')

    def update_path_preview(self):
        extension = self.ids.extension_spinner.text
        new_filter = ["*{}".format(extension)]
        self.ids.filechooser_save.filters = new_filter
        # path_preview = filename + extension
        # self.ids.path_preview.text = path_preview

    # on_selection: root.file_selection(filechooser_save.path, filechooser_save.selection)
    def file_selection(self, path, filename):

        print(path, filename)
        # extract the filename without the extension from the fullpath
        filename = filename[0].split('\\')[-1]
        print(filename)
        filename = filename.split('.')[0]
        self.ids.filename_input.text = filename
        # self.ids.path_preview.text = os.path.join(path, filename)

        return

    def save_file(self):
        path = self.ids.path_preview.text
        # Implement your file saving logic here, considering edge cases
        try:
            # ... (file saving code) ...
            self.model_item.save_prediction_in_excel_helper(path)
            self.dismiss()  # Close the popup on successful save
        except Exception as e:
            # Handle exceptions (e.g., invalid path, unsupported extension)
            # Display an error message to the user
            print(f"Error saving file: {e}")
            pass



class Input_Tabbed_Panel(TabbedPanel):

    def __init__(self, **kwargs):
        super(Input_Tabbed_Panel, self).__init__(**kwargs)
        # self.default_tab = TabbedPanelItem(text='Default Tab')
        # self.switch_to(self.default_tab)
        # Schedule the switch_to method using the Clock
        Clock.schedule_once(self.switch_to_tab, 0)

    def switch_to_tab(self, dt):
        # Switch to the first tab
        if len(self.tab_list) > 0:
            self.switch_to(self.tab_list[-1])

    def add_tab(self, tab_name, tab_data):
        ti = Input_Tabbed_Panel_Item(text=tab_name, tab_data=tab_data)
        # close_button = Button(text='X', size_hint=(None, None), size=(30, 30), pos_hint={'x': 0.9, 'y': 0.9})
        # close_button.bind(on_release= lambda x: self.on_close_button_tap(ti))
        # ti.add_widget(close_button)
        self.add_widget(ti)

    def load_inputs(self, inputs):
        for input in inputs:
            for tab in self.tab_list:
                if tab.text == input['text']:
                    tab.save_input(
                        input['val'], input['value_to_display'], input['isArray'])
                    break


class Input_Tabbed_Panel_Item(TabbedPanelItem):

    type_label = ObjectProperty(None)
    description_label = ObjectProperty(None)
    val = ObjectProperty(None)
    value_to_display = ObjectProperty(None)
    isArray = BooleanProperty(False)

    def __init__(self, text, tab_data, **kwargs):
        super(Input_Tabbed_Panel_Item, self).__init__(text=text, **kwargs)
        # self.text = tab_name
        print(tab_data)
        self.val = None
        self.type_label.text = "Type: " + tab_data['type']
        self.description_label.text = "Description: " + \
            remove_extra_spaces(tab_data['desc'])

    def save_input(self, input, to_display, isArray=False):
        self.val = input
        self.value_to_display = to_display
        self.isArray = isArray
        print("Saving Input Value : ", self.val)
        print("Saving Input Value to Display : ", self.value_to_display)
        print("Is Array : ", self.isArray)
        print()


class Select_Input_Screen(Screen):
    pass


class ExcelTreeView(TreeView):
    def __init__(self, **kwargs):
        super(ExcelTreeView, self).__init__(**kwargs)

        self.excel_file_path = None
        self.hide_root = True
        self.our_root = self.add_node(
            TreeViewLabel(text=str("Drop File Here"), is_open=False))
        # self.load_excel_data()

    def load_excel_data(self, excel_file_path):
        # Load Excel file
        self.excel_file_path = excel_file_path
        excel_data = pd.read_excel(self.excel_file_path, sheet_name=None)

        if self.our_root is not None:
            self.remove_node(self.our_root)

        self.our_root = self.add_node(
            TreeViewLabel(text=str(excel_file_path), is_open=True))

        # Populate the TreeView with sheets and columns
        for sheet_name, sheet_data in excel_data.items():
            sheet_node = self.add_node(
                TreeViewLabel(text=str(sheet_name), is_open=True), self.our_root)

            for column_name in sheet_data.columns:
                # Show column name and first 5 values in the column
                text_to_show = str(column_name) + " : " + \
                    str(sheet_data[column_name].head(4).tolist())

                this_column_data = {
                    'file_path': self.excel_file_path,
                    'sheet_name': sheet_name,
                    'column_name': column_name,
                }

                column_node = self.add_node(
                    TreeViewLabel(text=text_to_show), sheet_node)
                column_node.column_data = this_column_data

                # Bind the on_touch_down event to handle column click
                column_node.bind(on_touch_down=self.on_column_click)

    def load_csv_data(self, csv_file_path):

        self.excel_file_path = csv_file_path

        excel_data = pd.read_csv(self.excel_file_path)

        if self.our_root is not None:
            self.remove_node(self.our_root)

        self.our_root = self.add_node(
            TreeViewLabel(text=str(csv_file_path), is_open=True))

        # Open the CSV file and read the first line to get the column names

        for column_name in excel_data.columns:
            # Show column name and first 5 values in the column
            text_to_show = str(column_name) + " : " + \
                str(excel_data[column_name].head(4).tolist())
            this_column_data = {
                'file_path': self.excel_file_path,
                'sheet_name': 'CSV',
                'column_name': column_name,
            }

            column_node = self.add_node(
                TreeViewLabel(text=text_to_show), self.our_root)
            column_node.column_data = this_column_data

            # Bind the on_touch_down event to handle column click
            column_node.bind(on_touch_down=self.on_column_click)

    def on_column_click(self, instance, touch):
        if instance.collide_point(*touch.pos):
            # Handle the column click
            if touch.is_double_tap:

                # if self.excel_file_path.endswith(".xlsx"):
                #     sheet_name = instance.parent_node.text
                #     column_name = instance.text

                #     # Retrieve the column data using pandas
                #     excel_data = pd.read_excel(
                #         self.excel_file_path, sheet_name=sheet_name)
                #     column_data = excel_data[column_name].tolist()
                # elif self.excel_file_path.endswith(".csv"):
                #     sheet_name = "CSV"
                #     column_name = instance.text

                #     excel_data = pd.read_csv(self.excel_file_path)
                #     column_data = excel_data[column_name].tolist()

                # Print the whole column data as a list
                print(
                    f'Double Click" - Column: {instance.text}, Sheet: {instance.parent_node.text}')
                # print(self.excel_tree_view_header)

                this_column_data = instance.column_data
                print(this_column_data)

                column_name = this_column_data['column_name']

                # Disble button
                btn = Button(text='{0}'.format(column_name), size_hint_x=None, width=len(
                    column_name)*15 if len(column_name)*15 > 100 else 100)
                self.excel_tree_view_header.add_widget(btn)

                self.root_window.array_input_list.append(this_column_data)
                self.root_window.string_input_list.append(column_name)

            else:
                print(
                    f'Single Click - Column: {instance.text}, Sheet: {instance.parent_node.text}')
                # print(column_data)


class FileDropPopup(Popup):

    excel_file_path = ObjectProperty(None)

    array_input_list = ListProperty([])
    string_input_list = ListProperty([])

    def __init__(self, input_screen, **kwargs):
        super(FileDropPopup, self).__init__(**kwargs)
        self.input_screen = input_screen

        Window.bind(on_dropfile=self.on_drop_file)

    def on_drop_file(self, window, file_path):
        # Handle the dropped file in the popup
        # self.drop_label.text = f'Dropped File:\n{file_path}\nAt: ({x}, {y})'
        # print(f'Dropped File:\n{file_path}\nAt: ({x}, {y})')

        # self.ids.excel_tree_view.load_excel_data(file_path.decode("utf-8"))
        # self.load("",file_path.decode("utf-8"))
        self.excel_file_path = file_path.decode("utf-8")

        if self.excel_file_path.endswith(".xlsx"):
            self.ids.excel_tree_view.load_excel_data(self.excel_file_path)
        elif self.excel_file_path.endswith(".csv"):
            self.ids.excel_tree_view.load_csv_data(self.excel_file_path)

    def load(self, path, filename):

        # Handle error
        if path == '':
            return
        if len(filename) == 0:
            return
        
        
        # self.ids.left_file_drop_popup.remove_widget(self.ids.drop_file_label)
        self.excel_file_path = os.path.join(path, filename[0])

        if self.excel_file_path.endswith(".xlsx"):
            self.ids.excel_tree_view.load_excel_data(self.excel_file_path)
        elif self.excel_file_path.endswith(".csv"):
            self.ids.excel_tree_view.load_csv_data(self.excel_file_path)

        # self.ids.excel_tree_view.load_excel_data(self.excel_file_path)

    # on pop up close, unbind the on_drop_file event handler

    def on_dismiss(self):
        Window.unbind(on_drop_file=self.on_drop_file)
        print("Unbinding on_drop_file")
        return super().on_dismiss()

    def submit(self):
        print("Submit")
        print(self.input_screen.root)
        self.input_screen.root.save_input(
            self.array_input_list, str(self.string_input_list), True)
        self.dismiss()


class Array_Input_Screen(Screen):

    def open_file_drop_popup(self, instance):
        # Open the file drop popup
        file_drop_popup = FileDropPopup(input_screen=self)
        file_drop_popup.open()


class String_Input_Screen(Screen):

    pass


class Boolean_Input_Screen(Screen):

    pass


class Integer_Input_Screen(Screen):
    pass


class TopMenuBar(ActionBar):
    pass


class IntegerInput(TextInput):
    pattern = StringProperty("^[0-9]*$")

    def insert_text(self, substring, from_undo=False):
        pat = re.compile(self.pattern)
        if pat.match(substring):
            s = substring
            return super(IntegerInput, self).insert_text(s, from_undo=from_undo)
        else:
            return super(IntegerInput, self).insert_text('', from_undo=from_undo)


class Float_Input_Screen(Screen):
    pass


class FloatInput(TextInput):

    pat = re.compile('[^0-9]')

    def insert_text(self, substring, from_undo=False):
        pat = self.pat
        if '.' in self.text:
            s = re.sub(pat, '', substring)
        else:
            s = '.'.join(
                re.sub(pat, '', s)
                for s in substring.split('.', 1)
            )
        return super().insert_text(s, from_undo=from_undo)


class MessageBox(Popup):
    message = StringProperty("")
    background_color = ListProperty([1, 1, 1, 1])  # Default to white

    def __init__(self, title="", message="", message_type="info"):
        super(MessageBox, self).__init__(title=title, message=message)
        self.message = message
        # Assign a background color based on the message type

        self.background = 'white'
        if message_type == "error":
            self.background_color = [1, 0, 0, 1]  # Red
        elif message_type == "success":
            self.background_color = [0, 1, 0, 1]  # Green
        elif message_type == "info":
            self.background_color = [0, 0, 1, 1]  # Blue
        elif message_type == "warning":
            self.background_color = [1, 0.5, 0, 1]  # Orange


class Model_Tabbed_Panel(TabbedPanel):

    def __init__(self, **kwargs):
        super(Model_Tabbed_Panel, self).__init__(**kwargs)
        # self.default_tab = TabbedPanelItem(text='Default Tab')
        # self.switch_to(self.default_tab)

    def add_tab(self, tab_name, tab_data):

        if (self.main_widget.CURRENT_MODEL_RUNS_DIR is None):
            # Print in RED that there is no project open
            print("\033[91mThere is no project open\033[0m")
            msg = "There is no project open. Please open a project to continue."
            MessageBox(title="Error", message=msg,
                       message_type="error").open()
            return

        print("Adding Tab", tab_name, tab_data)
        ti = Model_Tabbed_Panel_Item(
            text=tab_name, tab_data=tab_data, main_widget=self.main_widget)
        # close_button = Button(text='X', size_hint=(None, None), size=(30, 30), pos_hint={'x': 0.9, 'y': 0.9})
        # close_button.bind(on_release= lambda x: self.on_close_button_tap(ti))
        # ti.add_widget(close_button)
        self.add_widget(ti)
        self.switch_to(ti, do_scroll=True)

    def on_close_button_tap(self, instance):
        print("User tapped on close button")
        print(instance)
        print(type(instance))
        self.remove_widget(instance)
        if len(self.tab_list) == 0:
            self.switch_to(self.default_tab)
        else:
            self.switch_to(self.tab_list[0], do_scroll=True)

        print("Main Widget", self.main_widget)

    def clear_tabs(self):
        self.clear_widgets()


class Left_TreeViewLabel(TreeViewLabel):
    pass


class Left_Excel_Panel(TreeView):

    def __init__(self, **kwargs):
        super(Left_Excel_Panel, self).__init__(**kwargs)

        # # Add a top node to the TreeView
        # self.root_options = dict(text='Statsmodels.api', is_open=True)
        self.hide_root = True
        data = pd.read_excel('Statsmodels_Functions.xlsx',
                             sheet_name='Statsmodels.api')

        root1 = self.add_node(
            Left_TreeViewLabel(text="Statsmodels.api", is_open=True))

        # Create a Statsmodels.

        self.stats_model_list = {}
        name = ''
        for idx, row in data.iterrows():
            if pd.notna(row['Name']):
                name = row['Name']
                self.stats_model_list[name] = {row['Function']: (
                    row['Type(categories)'], row['Text_Form'], row['Links'])}
            elif pd.notna(row['Function']):
                self.stats_model_list[name][row['Function']] = (
                    row['Type(categories)'], row['Text_Form'], row['Links'])

        # print(self.stats_model_list)

        for key in self.stats_model_list:
            parent = self.add_node(
                Left_TreeViewLabel(text=str(key), is_open=True), root1)
            # Create a
            for k in self.stats_model_list[key]:
                leaf_node = Left_TreeViewLabel(text=str(k), is_leaf=True)
                self.add_node(leaf_node, parent)

                # Bind on_touch_down event for leaf nodes
                leaf_node.bind(on_touch_down=self.on_leaf_node_tap)

        current_dir = os.getcwd()
        analytical_folder = os.path.join(current_dir, "Analytic")

        # Populate the TreeView with directories and files
        root2 = self.add_node(
            Left_TreeViewLabel(text=os.path.basename(analytical_folder), is_open=True))

        self.populate_tree_view(self, root2, analytical_folder, "Analytic")

    def populate_tree_view(self, tree_view, parent_node, directory, parent_name):
        # Get the list of files and directories in the current directory
        files_and_dirs = os.listdir(directory)

        # Iterate over the files and directories
        for file_or_dir in files_and_dirs:
            # Get the full path of the file or directory
            full_path = os.path.join(directory, file_or_dir)

            # Check if the current item is a directory
            if os.path.isdir(full_path):
                # Create a TreeViewLabel for the directory
                dir_node = Left_TreeViewLabel(text=file_or_dir, is_open=True)

                # Add the directory node to the parent node
                tree_view.add_node(dir_node, parent_node)

                # Recursively populate the TreeView with the contents of the directory
                self.populate_tree_view(
                    tree_view, dir_node, full_path, parent_name + "." + file_or_dir)
            else:
                # Create a TreeViewLabel for the file
                if file_or_dir.endswith(".py"):
                    file_node = Left_TreeViewLabel(
                        text=file_or_dir, is_leaf=True)

                    # Add the file node to the parent node
                    tree_view.add_node(file_node, parent_node)

                    filename_without_extension = os.path.splitext(file_or_dir)[
                        0]

                    print("Parent Name : ", parent_name +
                          "." + file_or_dir + "." + file_or_dir)
                    # Bind the on_touch_down event to handle file click
                    file_node.bind(on_touch_down=lambda instance,
                                   touch: self.on_leaf_node_tap_analytic(instance, touch, parent_name + "." + filename_without_extension + "." + filename_without_extension))

    def on_leaf_node_tap_analytic(self, instance, touch, import_path):
        if instance.collide_point(*touch.pos):
            print("User tapped on", instance.text)

            # Get the full path of the file
            # ('OLS(endog[,\xa0exog,\xa0missing,\xa0hasconst])', 'Ordinary Least Square', 'https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS')

            tab_data = (instance.text, instance.text, "Analytic#"+import_path)
            print(tab_data)

            # Add the tab to the Model Tabbed Panel
            self.model_tabbed_panel.add_tab(instance.text, tab_data)

            return True

    def on_leaf_node_tap(self, instance, touch):
        if instance.collide_point(*touch.pos):

            # print("User tapped on", instance.text)
            # # app = App.get_running_app()
            tp = self.model_tabbed_panel
            # # tp.clear_tabs()
            tp.add_tab(
                instance.text, self.stats_model_list[instance.parent_node.text][instance.text])

            # ti = TabbedPanelItem(text=instance.text, content=BoxLayout(orientation="vertical"))

            # ti.content.add_widget(Label(text=self.stats_model_list[instance.parent_node.text][instance.text][2]))
            # ti.content.add_widget(Label(text=self.stats_model_list[instance.parent_node.text][instance.text][2]))

            # close_button = Button(text='X', size_hint=(None, None), size=(30, 30))
            # close_button.bind(on_release= lambda x: self.on_close_button_tap(ti))
            # ti.content.add_widget(close_button)

            # tp.add_widget(ti)

            # print(self.stats_model_list[instance.text])

            return True


class CreateNewProjectPopup(Popup):

    def __init__(self, main_widget, save, **kwargs):
        super(CreateNewProjectPopup, self).__init__(**kwargs)
        self.main_widget = main_widget
        self.to_save = save
        # Sample project name with timestamp
        self.ids.project_name_input.text = "Project_Name_" + \
            pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M-%S')
        if save:
            self.ids.save_button.text = "Save"
        else:
            self.ids.save_button.text = "Create"



    def create_project(self):

        project_name = self.ids.project_name_input.text
        if project_name == "":
            return

        project_dir = self.ids.filechooser_save.path

        if project_dir == "":
            return

        print("Creating Project", project_name, project_dir)

        # Create a new folder with the project name given by the user.
        project_folder = os.path.join(project_dir, project_name)

        if not os.path.exists(project_folder):
            os.mkdir(project_folder)
            print("Project Folder Created")
        else:
            print("Project Folder Already Exists")
            return

        self.main_widget.CURRENT_PROJECT_NAME = project_name
        self.main_widget.CURRENT_PROJECT_DIR = project_folder
        self.main_widget.CURRENT_MODEL_RUNS_DIR = os.path.join(
            project_folder, "MODEL_RUNS")

        self.main_widget.ids.action_previous.text = project_name
        print("Project Folder", project_folder)
        print("self.main_widget.ids.action_previous.text",
              self.main_widget.ids.action_previous.text)

        if not os.path.exists(self.main_widget.CURRENT_MODEL_RUNS_DIR):
            os.mkdir(self.main_widget.CURRENT_MODEL_RUNS_DIR)
            print("Model Runs Folder Created")
        else:
            print("Model Runs Folder Already Exists")

        self.dismiss()
        if self.to_save:
            self.main_widget.save_project(project_folder)

        # Show a message box that the project has been created
        MessageBox(title="Success", message="Project Created Successfully",
                     message_type="success").open()




class OpenProjectPopup(Popup):

    def __init__(self, main_widget, **kwargs):
        super(OpenProjectPopup, self).__init__(**kwargs)
        self.main_widget = main_widget

    def open_project(self):

        project_dir = self.ids.filechooser_save.path

        if project_dir == "":
            return

        project_name = project_dir.split('\\')[-1]

        print("Opening Project", project_name, project_dir)

        self.main_widget.CURRENT_PROJECT_NAME = project_name
        self.main_widget.CURRENT_PROJECT_DIR = project_dir
        self.main_widget.CURRENT_MODEL_RUNS_DIR = os.path.join(
            project_dir, "MODEL_RUNS")

        self.main_widget.ids.action_previous.text = project_name

        self.main_widget.open_project(project_dir)

        self.dismiss()

def format_array_value(value):
    if isinstance(value, list) and value and isinstance(value[0], dict):
        # Assuming all items in the list are dictionaries with the same structure.
        formatted_list = []
        for item in value:
            # Format each dictionary into a readable string.
            item_str = f"File: {os.path.basename(item['file_path'])}, Sheet: {item['sheet_name']}, Column: {item['column_name']}"
            formatted_list.append(item_str)
        # Join all strings into one multi-line string.
        return '\n'.join(formatted_list)
    return value


def format_array_value_with_pretty_table(value):
    if isinstance(value, list) and value and isinstance(value[0], dict):
        formatted_tables = []

        table = PrettyTable()
        table.field_names = ["File Path", "Sheet Name", "Column Name"]
        table.align = "l"
        for item in value:
            # Create a PrettyTable for each dictionary in the list

            # Adding rows for each key-value pair in the dictionary
            table.add_row([item['file_path'], item['sheet_name'], item['column_name']])

        return table.get_string()
    return value


class MainWidget(Widget):

    # file_options_list = ListProperty(["New Account", "New Global Template", "New Local Template", "New Project from Analytical Folder",
    #                                   "New Project from Template", "New Run from Template", "New Run from Project", "Open Project", "Open Run", "Save Project", "Save Run"])

    # Analytical_Folder_dir = os.path.join(os.getcwd(), "Analytic")
    # print(Analytical_Folder_dir)

    # self.
    CURRENT_PROJECT_NAME = None
    CURRENT_PROJECT_DIR = None
    CURRENT_MODEL_RUNS_DIR = None

    def openCreateNewProjectPopup(self):
        create_new_project_popup = CreateNewProjectPopup(self, save=False)
        create_new_project_popup.open()

    # def save_project_popup(self):

    #     save_project_popup = SaveProjectPopup(self)
    #     save_project_popup.open()

    def save_as_project(self):
        print("Save As Project")

        if self.CURRENT_PROJECT_NAME is None:
            # Print in RED that there is no project open
            print("\033[91mThere is no project open\033[0m")
            msg = "There is no project open. Please open a project to continue."
            MessageBox(title="Error", message=msg,
                          message_type="error").open()
            return
        
        create_new_project_popup = CreateNewProjectPopup(self, save=True)
        create_new_project_popup.open()


    def save_project(self, fullDir_path):
        if self.CURRENT_PROJECT_NAME is None:
            print("\033[91mThere is no project open\033[0m")
            msg = "There is no project open. Please open a project to continue."
            # Assuming MessageBox is defined elsewhere
            MessageBox(title="Error", message=msg, message_type="error").open()
            return

        fullpath_json = os.path.join(
            fullDir_path, self.CURRENT_PROJECT_NAME + "_log.json")
        fullpath_txt = os.path.join(
            fullDir_path, self.CURRENT_PROJECT_NAME + "_summary.txt")
        data_to_save = []

        print("Saving Project in", fullpath_json)

        summary_tables = []  # For storing pretty table strings

        for model in self.ids.model_tabbed_panel.tab_list:  # Assumed model structure
            data_for_model = {'text': model.text,
                              'tab_data': model.tab_data, 'inputs': []}

            model_table = PrettyTable()
            model_table.field_names = ["Input Name",
                                       "Value", "Display Value", "Is Array"]
            model_table.title = f"Model: {model.text}"

            for input in model.input_tabbed_panel.tab_list:  # Assumed input structure
                formatted_value = format_array_value_with_pretty_table(
                    input.val)
                data_for_model['inputs'].append({
                    'text': input.text,
                    'val': input.val,  # Original value
                    'value_to_display': input.value_to_display,
                    'isArray': input.isArray
                })
                model_table.add_row(
                    [input.text, formatted_value, input.value_to_display, "Yes" if input.isArray else "No"])

            data_to_save.append(data_for_model)
            summary_tables.append(model_table.get_string())

        with open(fullpath_json, 'w') as f:
            json.dump({"models": data_to_save}, f, indent=4)

        # Reverse the summary Table
        summary_tables = summary_tables[::-1]
        
        with open(fullpath_txt, 'w') as f_txt:
            f_txt.write("\n\n".join(summary_tables))

        MessageBox(title="Success", message="Project Saved Successfully",
                   message_type="success").open()

    def open_project_popup(self):
        open_project_popup = OpenProjectPopup(self)
        open_project_popup.open()

    def open_project(self, fullDir_path):

        fullpath = os.path.join(
            fullDir_path, self.CURRENT_PROJECT_NAME + "_log.json")

        with open(fullpath, 'r') as f:
            data_to_load = json.load(f)

        data_to_load = data_to_load['models']
        # Clear the current tabs
        models = self.ids.model_tabbed_panel.tab_list[::-1]
        for model in models:
            model.close_button.dispatch("on_press")

        for model in data_to_load[::-1]:
            self.ids.model_tabbed_panel.add_tab(
                model['text'], model['tab_data'])
            self.ids.model_tabbed_panel.tab_list[0].input_tabbed_panel.load_inputs(
                model['inputs'])
            # for input in model['inputs']:
            #     # self.ids.model_tabbed_panel.tab_list[-1].input_tabbed_panel.add_tab(
            #     #     input['text'], input)
            #     print(input['text'], input['val'], input['value_to_display'], input['isArray'])

        # Save the project in the fullpath
        # Save the Models, for each model, the inputs(if array, then the file path, sheet name and column name will be saved), the prediction files and the summaries in a file in the project folder.
        # Save the Models

    # def file_options_selected(self, options):

    #     print("User selected", options)
    #     self.ids.file_options_spinner.text = "File"

    # def file_type_entered(self, text):
    #     print("User entered", text)

    # def on_Analytical_selection(self, selection):
    #     print("Analytical Folder selection changed", selection)
    #     if len(selection) == 0:
    #         return
    #     file_name = selection[0]

    #     if file_name.endswith(".py"):
    #         execute_code_from_file(file_name)
    #         print("Done with", file_name)


class TemplateApp(App):
    def build(self):
        self.main_widget = MainWidget()
        return self.main_widget


if __name__ == '__main__':
    temp = TemplateApp()
    temp.run()

    # print(temp.main_widget.ids.model_tabbed_panel.tab_list[0].ids.model_name_label.text)
    # for i in temp.main_widget.ids.model_tabbed_panel.tab_list:
    #     print(i.ids.model_name_label.text)


'''

After opening the software, the user will be able to add models to the GUI:
User can also add inputs to the models.
If the inputs are of type String, Integer, Float, Boolean, the user can directly enter the value and only one input will be saved.
if the inputs are of type Array, the user can select the excel file or csv file. The sheets and columns name with 4 sample data will be shown in the tree.
User can select the columns and add them to the input list. The file path, sheet name and column name will be saved in the input list for each column selected.
But user cannot run any model until he create a new project or open an existing project.
Createing a project simply means creating a new folder with the project name given by the user.
The project folder will have 1 sub folders: MODEL_RUNS
Now if the user wants to run a model, he can run it using the run button in the model tab. The model prediction will be saved in the MODEL_RUNS folder with the model name and the date and time of the run.
If the user want to use this prediction in another model, he can do so by selecting the prediction file from the MODEL_RUNS folder.
After doing all the runs, the user can also save the project. Save the project will save the Models, for each model, the inputs(if array, then the file path, sheet name and column name will be saved), the prediction files and the summaries in a file in the project folder.
When the user opens the project later, he can see all the models tabs loaded with the inputs(if array, then the file path, sheet name and column name is restored, which implies that the updated data will be used for the prediction),and the summaries.
User can also save the current project as a template. Which will only save the models list used in the project.
User can also open a template, which will load the models in the GUI, but the inputs and the summaries will not be loaded.


'''
