Backend_prompt = {
    "1": """
    Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks. 
Now, the drafter is trying to insert a table on the target area boxed by rectangle markups. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the content of a table) from high to low. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks to complete the execution of the instruction. Don't assume the value of missing necessary information, but identify tasks lacking necessary information, record the missing details, and execute the task that all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
0.1     Import libraries
1.	Open the file.
2.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
3.	Modify the task instructions into dictionaries. Each dictionary contains the arguments for a table-adding mission. 
4.	Argument collection: Prepare parameters for adding a table by modifying the argument's value specified by the drafter's instructions into a dictionary.
5.	Table adding: Create a table manipulator instance and add the table.
6.	Save the file you have opened as a new file with an updated name.


Available tools:
Two constants defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;


Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the text manipulator instance. Here is an example dictionary including all possible attributes:
argument = {“clip”: a rectangle refers to the target position; “data”: a list contains the content of each row, general form: [[row1 content],[row2 content], ..., [ith row content]]; “arrange”: a list of row height and column width, general form:[[row height],[column width]], if only row height is defined, it is noted as [[row height], None], and vice versa as [None,[column width]].; “font”: a string refers to the font of the text; “fontsize”: an int refers to the size of text }
The drafter will selectively specify parameters such as font, font size etc. as needed, so please create a dictionary for each task that matches the description. 

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"

Table adding:
A table adder is instantiated by specifying the essential arguments (document(doc), page number(pagenumber), clip, data) and mentioned optional arguments (arrange, font, fontsize) on demand like this:
“Table_adder = PDFbf.manipulate_table(doc=fitz.document, pagenumber=int, clip=rectangle, data=list, arrange=list, font=string, fontsize=int)” 
You can get a doc with a table added by calling the following function:
“doc = Table_adder.addtable()” 

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the missing information, the standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: the relevant part of the original instruction, missing if there is no such part, fill in the relevant part of the original instruction and mark it as 'incomplete' in brackets if there is a relevant part but not enough necessary information'''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"


Here is an example:

User: The file name is “af35dsdd_table.pdf”. Add tables in the first box on the second page. The first table: table content is: first row: "A, B, C", second row: "1, 2, 3", third row: "X, Y, Z". Font size 12. Row height: [30, 18, 18].The second table: table content is: first row: "Name, Age, Gender", second row: "Alice, 30, F", third row: "Bob, 25, M". Row height: [20, 15, 15]. Column width 50.
Add tables in the second box on page 4. The first table content is: first row: "Item, Price", second row: "Apple, 1.0", third row: "Banana, 0.5". Font size 14. Column width: [20, 20]. The second table content is: first row: "Date, Time", second row: "2023-01-01, 12:00", third row: "2023-01-02, 13:00". Row height 20, column width: 60. The third table content is:

Response:
```python

import fitz
import PDFbf

#Task 1
#file 1
filepath = "af35dsdd_table.pdf"
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc=doc)

#page 1
pagenumber0 = 1
#rectangle 1
rectangleorder0 = 0
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
#table 1
argument0 = dict(clip=rectangle0, data=[["A", "B", "C"], ["1", "2", "3"], ["X", "Y", "Z"]], arrange=[[30, 18, 18], None], fontsize=12)
table_adder0 = PDFbf.manipulate_table(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), data=argument0.get("data"), arrange=argument0.get("arrange"), fontsize=argument0.get("fontsize"))
doc = table_adder0.addtable()
#Task 2
#table 2
argument1 = dict(clip=rectangle0, data=[["Name", "Age", "Gender"], ["Alice", "30", "F"], ["Bob", "25", "M"]], arrange=[[20, 15, 15], [50, 50, 50]])
table_adder1 = PDFbf.manipulate_table(doc=doc, pagenumber=pagenumber0, clip=argument1.get("clip"), data=argument1.get("data"), arrange=argument1.get("arrange"))
doc = table_adder1.addtable()

#Task 3
#page 2
pagenumber1 = 3
#rectangle 2
rectangleorder1 = 1
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)
#table 3
argument2 = dict(clip=rectangle1, data=[["Item", "Price"], ["Apple", "1.0"], ["Banana", "0.5"]], arrange=[None,[20, 20]], fontsize=14)
table_adder2 = PDFbf.manipulate_table(doc=doc, pagenumber=pagenumber1, clip=argument2.get("clip"), data=argument2.get("data"), arrange=argument2.get("arrange"), fontsize=argument2.get("fontsize"))
doc = table_adder2.addtable()
#Task 4
#table 4
argument3 = dict(clip=rectangle1, data=[["Date", "Time"], ["2023-01-01", "12:00"], ["2023-01-02", "13:00"]], arrange=[[20, 20, 20], [60, 60]])
table_adder3 = PDFbf.manipulate_table(doc=doc, pagenumber=pagenumber1, clip=argument3.get("clip"), data=argument3.get("data"), arrange=argument3.get("arrange"))
doc = table_adder3.addtable()

# Task 5
# Recording the missing information for Task 5 (not enough details for table adding)
missing_information1 = '''file-level:af35dsdd_table.pdf, page-level: 4, order-level: 2, base-level: missing'''

recorder1 = PDFbf.recorder(
    missing_information=missing_information1
)
recorder1.recording()

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```
    """,
    "2": """
    Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks. 
Now, the drafter is trying to edit the table content on the target area boxed by rectangle markups. Editing a table on a target position is seen as a task. There are three possible operations that must be selected from: deleting, clearing, and replacing. There are also format attributes that can be optionally specified on demand: the font, font size, alignment, and border width. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the necessary information of the operation) from high to low. The first three levels of information determine the target position, while the base level describes the operation. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. You need to find the four level information corresponding to each task from an instruction that may contain multiple tasks. Don't assume its value for necessary information if the command doesn't mention it or describe it vaguely. For all the tasks in commands, execute the tasks where all essential information is completely specified, while for tasks lacking necessary information, record the missing details. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information (file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for updating text content by modifying the argument's value specified by the drafter's instructions into a dictionary. 
5.	Extract original table: Create a table extractor instance and extract the original table.
6.	Update table: Create a table manipulator instance, revise the table content, and add the new table.
7.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the table editing task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the table that needed to be revised; "font": a string refers to the font of the text in table; "fontsize":an int refers to the size of text in table; "borderwidth": an int refers to the width of the table border, "align": a string refers to the align of the text in table, which can be one of "center", "left", "right"}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"


Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"


Extract the original table:
A table extractor can be instantiated by specifying the document(doc), page number(pagenumber), clip like this:
"table_extractor = PDFbf.extract_table(doc=fitz.document, pagenumber=int, clip=rectangle)"
You can get the date of the table by calling the following function:
"table_data = table_extractor.data" 

Update table:
A table manipulator can be instantiated by specifying the essential arguments (document(doc), page number(pagenumber), clip, data) and optional arguments (font, fontsize, borderwidth, align) on demand like this:
"table_manipulator=PDFbf.manipulate_table(doc=fitz.document, pagenumber=int, clip=rectangle, data=list, font=string, fontsize=int, borderwidth=int, align=string)" 

Here are some arguments that should be specified when you want to delete specified contents from the original tables:
"delcolumn": a list of columns that need to be deleted, [1] for the first column;
"delrow": similar to "delcolumn", a list of rows that need to be deleted, [5] for the fifth row;

You can get the revised table data by deleting the whole table or specified row or column from the original table data by specifying rows to delete(delrow), columns to delete(delcolumn) and calling the following function:
"table_manipulator.data = table_manipulator.cuttable(delrow=list,delcolumn=list)" 
"table_manipulator.data = table_manipulator.cuttable()" delete the whole table if the delete table action is required with no target rows and columns specified.
"table_manipulator.data = table_manipulator.cuttable(delrow=[3,4,5,6])" delete the 3rd to 6th rows from the original table.
"table_manipulator.data = table_manipulator.cuttable(delrow=[3],delcolumn=[2,4])" Delete the 3rd row and delete the 2nd and 4th columns from the original table.

Here are some arguments that should be specified when you want to clear or replace specified contents from the original tables:
"startcell": a list refers to the position of the start cell to be cleared or replaced in the data list, general form: [row number, column number];
"endcell": a list refers to the position of the end cell to be cleared or replaced in the data list, general form: [row number, column number];
"repdata": a list of data to replace the specified content from original table, general form: [[row1 content],[row2 content],...,[ith row content]];

You can get the revised table data by clearing the whole table or the data of specified cells of the original table by defining the start cell(startcell), end cell(endcell) and calling the following function:
"table_manipulator.data = table_manipulator.emptytable(startcell=list,endcell=list)"
"table_manipulator.data = table_manipulator.emptytable()" clear the whole table if the clear table action is required with no start cell(startcell) and end cell(endcell) specified.
"table_manipulator.data = table_manipulator.emptytable(startcell=[2,3],endcell=[5,7])" clear the content of cells starting from the 2nd row and 3rd column, end at the 5th row and 7th column from the original table.
“table_manipulator.data = table_manipulator.emptytable(startcell=[“all”,3])” clear the column 3.
“table_manipulator.data = table_manipulator.emptytable(startcell=[7,“all”])” clear the row 7.
"table_manipulator.data = table_manipulator.emptytable(startcell=[8,9])" clear the data of one cell at the 8th row and 9th column from the original table if clear table action is required with only one cell specified.

You can get the revised table data by replacing the whole table or the data of specified cells from the original table by defining the data used to replace(repdata), start cell(startcell), end cell(endcell) and calling the following function:
"table_manipulator.data = table_manipulator.modifytable(repdata=replace_data, startcell=list, endcell=list)"
"table_manipulator.data = table_manipulator.modifytable(repdata=replace_data)" Replace the whole table with the replace data if no start cell(startcell) and end cell(endcell) are specified.
"table_manipulator.data = table_manipulator.modifytable(repdata=replace_data, startcell=[7,9], endcell=[12,13])" Replace the data of cells starting from 7th row and 9th column, end at the 12th row and 13th column from the original table with the replace data.
“table_manipulator.data = table_manipulator.emptytable(repdata=replace_data, startcell=[“all”,3])” replace the column 3.
“table_manipulator.data = table_manipulator.emptytable(repdata=replace_data, startcell=[7,“all”])” replace the row 7.
"table_manipulator.data = table_manipulator.modifytable(repdata=replace_data, startcell=[2,3])" Replace the data of the cell at 2nd row and 3rd column from the original table with the replace data if replace table action is required with only one cell specified.

You can get a doc with a revised table added by calling the following function:
"doc = table_manipulator.addtable()"

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no related part) or related part in the original instruction following with an "incomplete" in brackets (if there is some related description but some necessary information is missing)'''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"


Here is an example. You should respond in the same way.

User: The file name is "De222lsfqweete_table.pdf". 
For the table in the first box of page 4, clear the cells from the second row and third column to the fourth row and fifth column. Replace the cells from the third row and first column to the fourth row and second column with the data "first row: A, B; second row: C, D". Then delete the first row and the second to fifth columns. 
For the table on page 7, delete the second column. Clear the table.
For the seventh box on page eight, replace the table the data "first row: 10,11, second row: 25,26, third row: 31,32", clear the cell at 3rd row and 4th column, delete the 1-7th row and 10th columns. 

Response:
```python

import fitz
import PDFbf


#Task 1
#File 1
filepath = "De222lsfqweete_table.pdf"
doc = fitz.open(filepath)
annoextractor = PDFbf.extractanno(doc=doc)

#Page 1
pagenumber0 = 3
#Rectangle 1
rectangleorder0 = 0
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
#Operation 1
argument0 = dict(clip=rectangle0)
table_extractor0 = PDFbf.extract_table(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"))
table_data0 = table_extractor0.data
table_manipulator0 = PDFbf.manipulate_table(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), data=table_data0)
table_manipulator0.data = table_manipulator0.emptytable(startcell=[2, 3], endcell=[4, 5])
repdata0 = [["A", "B"], ["C", "D"]]
table_manipulator0.data = table_manipulator0.modifytable(repdata=repdata0, startcell=[3, 1], endcell=[4, 2])
table_manipulator0.data = table_manipulator0.cuttable(delrow=[1], delcolumn=[2,3,4,5])
doc = table_manipulator0.addtable()

#Task 2
#Lacking necessary information
missing_information1='''file-level: De222lsfqweete_table.pdf, page-level:7, order-level: missing, base-level: delete the second column. Clear the table.'''
recorder1 = PDFbf.recorder(missing_information=missing_information1)
recorder1.recording


#Task 3
#Page 3
pagenumber2=7
#Rectangle 3
rectangleorder2=6
rectangle2, rfpoint2 = annoextractor.getclip_rfpoint(pagenumber=pagenumber2,rectangleorder=rectangleorder2)
argument2 = dict(clip=rectangle2)
table_extractor2 = PDFbf.extract_table(doc=doc,pagenumber=pagenumber2,clip=argument2.get("clip"))
table_data2 = table_extractor2.data
table_manipulator2 = PDFbf.manipulate_table(doc=doc, pagenumber=pagenumber2, clip=argument2.get("clip"), data=table_data2)
repdata2 = [[10,11],[25,26],[31,32]]
table_manipulator2.data=table_manipulator2.modifytable(repdata=repdata2)
table_manipulator2.data = table_manipulator2.emptytable(startcell=[3,4])
table_manipulator2.data = table_manipulator2.cuttable(delrow=[1,2,3,4,5,6,7],delcolumn=[10])
doc = table_manipulator2.addtable()

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```
    """,
    "3": """
Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks.
Now, the drafter wants to change the position of a table on the target area boxed by rectangle markups. There are three operations that the drafter may want: translation, rotation, and scaling. The details defining the operation include operation type, direction, and magnitude. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the details for at least one of the three operations) from high to low. What needs to be made more clear is that the details defining each type of operation that can be considered as necessary information are: pairs of information consisting of the type of operation and the direction, translation-direction, rotation-direction, and scale-bigger or smaller, respectively. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks to complete the execution of the instruction. Don't assume the value of missing necessary information (only the magnitude of operation can be assumed), but identify tasks lacking necessary information, record the missing details, and execute the task that all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for changing a table position by modifying the argument's value specified by the drafter's instructions into a dictionary. 
5.	Changing position: Create a table projector instance and change position.
6.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the table position-changing task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the text that need to be dealt with; "move": a list refers to translation details, general form: [move in its right direction (positive value if move right, negative value if move left), move in its upper direction (positive value if move up, negative value if move down), reference_point], "rotation": a list refers to the rotation details, general form: ['r', rotation degrees in the clockwise direction (positive value in clockwise direction, negative value in counterclockwise direction),reference_point], if the direction of rotation is not specified, clockwise is considered; "scal": a list refers to the scaling, general form: [‘sc’, [zoom factor X direction, zoom factor Y direction], reference_point]}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"

Changing position:
A projector is instantiated by specifying the essential arguments (document(doc), page number(pagenumber), clip) and optional arguments (movement of translation (move), rotation(rotation) and scaling(scal)) like this:
"table_projector=PDFbf.Projector(doc=fitz.document,pagenumber=int,clip=rectangle,move=[move_in_right_direction, move_in_up_direction, reference_point],rotation=['r', rotation degrees in clockwise direction,reference_point],scal=['sc',[zoom factor in X direction, zoom factor in Y direction], reference_point])"
You can get a doc with a table mapped by calling following functions:
"doc = table_projector.project()" 

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the missing information, the standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: the relevant part of the original instruction, missing if there is no such part, fill in the relevant part of the original instruction following an 'incomplete' in brackets if there is a relevant part but not enough necessary information'''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an examples. You should respond in the same way.

User: The file name is "47_mapqweqping_table.pdf". Move the table in the first box on page 5 20 units left,30 units up. Rotate another table 45 degrees clockwise. Move the table in the second box of the tenth page 10 units left, 20 units down.
Response:
```python
import fitz
import PDFbf

#Task 1
#File 1
filepath = "47_mapqweqping_table.pdf"
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc=doc)


# page 5
pagenumber0 = 4
#rectangle 1
rectangleorder0 = 0
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
#Mapping 1
argument0 = dict(clip=rectangle0, move=[-20, 30, rfpoint0])
table_project0 = PDFbf.Projector(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), move=argument0.get("move"))
doc = table_project0.project()

#Task 2
# Recording the missing information for Task 2 (not enough details for rotating a table)
missing_information1 = '''file-level:47_mapqweqping_table.pdf, page-level: missing, order-level: missing, base-level: rotate 45 degrees clockwise'''

recorder1 = PDFbf.recorder(
    missing_information=missing_information1
)
recorder1.recording()

#Task 3
#Page 10
pagenumber1 = 9
#rectangle 2
rectangleorder1 = 1
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)
# Mapping 3
argument2 = dict(clip=rectangle1, move=[-10, -20, rfpoint1])
table_project2 = PDFbf.Projector(doc=doc, pagenumber=pagenumber1, clip=argument1.get("clip"), move=argument1.get("move"))
doc = table_project2.project()

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```


    """,
    "4": """
Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks. 
Now, the drafter wants to update the table format on the target area boxed by rectangle markups. The possible format attributes are: arrange, font, font size, alignment, border width. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing atte least one of the format attributs) from high to low. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks. Don't assume the value of missing necessary information (only the value of mentioned format attributes can be assumed), but identify tasks lacking necessary information, record the missing details, and execute the task that all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for updating text format by modifying the argument's value specified by the drafter's instructions into a dictionary.
5.	Table extraction: Create a table extractor instance and extract the original table..
6.	Table update: Create a table manipulator and update the original table format.
7.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the table format update task. Here is an example dictionary including all possible attributes:
{“clip”: a rectangle refers to the region of text should be dealing with. “arrange”: a list of row height and column width, general form:[[row height],[column width]], if only row height is defined, it is noted as [[row height], None], and vice versa as [None,[column width]]; “font”: a string refers to the font of the text; “fontsize”: an int refers to the size of text, “borderwidth”: an int refers to the width of the table border, “align”: a string refers to the align of the data in the table, which can be one of “center”, “left”, “right”, "justify";}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"

Table extracttion:
A table extractor can be instantiated by defining the document(doc), page number(pagenumber), clip like this:
“table_extractor = PDFbf.extract_table(doc=fitz.document, pagenumber=int, clip=rectangle)”
You can get a list of each row date of the table by calling following function:
“table_data = table_extractor.data”

Table update:
A table manipulator can be instantiated by defining the necessary arguments (document(doc), page number(pagenumber), clip, data) and optionally mentioned arguments (arrange, font, fontsize, borderwidth, align) like this:
“Table_manipulator = PDFbf.manipulate_table(doc=fitz.document, pagenumber=int, clip=rectangle, data=list, arrange=list, font=string, fontsize=int, borderwidth=int, align=string)” 
You can get a doc with the table updated by calling the following functions:
“doc = Table_manipulator.addtable()”

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no realted part) or related part in the original instruction following with a "incomplete" in brackets (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an example. You should respond in the same way to execute the user's instructions.
User: The file name is “2anipusdflate_table.pdf”. For the table in the third box on page two, change the align to the right. For the table on page one, change the font to "Times-Roman". 
Response:
```python

import fitz
import PDFbf

#Task 1
#file 1
filepath = “2anipusdflate_table.pdf”
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc=doc)

#page 2
pagenumber0=1
#rectangle 3
rectangleorder0=2
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
#table 1
argument0 = dict(clip=rectangle0,align="right")
table_extractor0 = PDFbf.extract_table(doc=doc,pagenumber=pagenumber0,clip=argument0.get("clip"))
table_data0 = table_extractor0.data
Table_manipulator0 = PDFbf.manipulate_table(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), data=table_data0, align=argument0.get("align"))
doc = Table_manipulator0.addtable()

#task 2 record the missing information
missing_information1 = '''file-level:2anipusdflate_table.pdf, page-level: 1, order-level: missing, base-level: change the font to "Times-Roman"'''
recorder1 = PDFbf.recorder(missing_information=missing_information1)
recorder1.recording

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```
    """,
    "5": """Text Adding instruction.
Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks. 
Now, the drafter wants to insert text on the target position boxed by rectangle markups. One string of text is seen as one task. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the text string) from high to low. A high-level information can correspond to multiple low-level information. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks to complete the execution of the instruction. Don't make assumptions if the previous four necessary information are missing, but you can assume a rational value according to your knowledge if other information is missing. Don't assume the value of necessary information, but identify tasks lacking necessary information, record the missing details, and execute the task only once all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.


Task plan:
The following is a standard process for completing a task:

0.	Incomplete information recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the remaining operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.	
4.	Argument collection: Prepare parameters for adding a text string by modifying the argument's value specified by the drafter's instructions into a dictionary.
5.	Text adding: Create a text manipulator instance and add text.
6.	Save the file you have opened as a new file with an updated name.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the text manipulator instance. Here is an example dictionary including all possible attributes:
argument = {"Clip": extracted target rectangle; "text": a string of new texts, "font": a string refers to the font of the text; "fontsize": an int refers to the font size; "textcolor": a string refers to the color of the text; "rotate": an int in 90, 180, 270, refers to the rotation of the text, "align": a fitz_constant selected from ```fitz.TEXT_ALIGN_RIGHT``` or ```fitz.TEXT_ALIGN_LEFT``` or ```fitz.TEXT_ALIGN_CENTER``` or ```fitz.TEXT_ALIGN_JUSTIFY``` refers to the align of the text}
The drafter will selectively specify parameters such as font, font size, text color, rotation, alignment, etc. as needed, so please create a dictionary for each task that matches the description. 

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"


Text adding:
A text manipulator is instantiated by specifying the essential arguments (document(doc), pagenumber, clip, text) and mentioned optional arguments (font, fontsize, text color, rotate, align). Here is an example with all possible arguments are mentioned:
"Text_manipulator=PDFbf.manipulate_text(doc=fitz.document, pagenumber=int, clip=rectangle, text=string, font=string, fontsize=int, textcolor=string, rotate=int, align=fitz_constant)" 
You can get a doc with texts added by calling the following function:
"doc=Text_manipulator.addtext()"

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no realted part) or related part in the original instruction following with an "incomplete" in brack (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an example and you should respond in a similar way.
User: The file name is "a123gfsdfd_text.pdf". In the second box of page three, add the text "Project Alpha" with font "times-roman", align left, and add text "Phase 1" with font "helv", rotation 0. In the fourth box of page five, add text with font courier and add the text "Approved" with text color blue, font size 14, align justify.

Response:
```pythons

import fitz
import PDFbf

#file 1 
filepath = "a123gfsdfd_text.pdf"
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc=doc)

# page 3
pagenumber0=2

# rectangle 2
rectangleorder0=1
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
# text string 1
argument0 = dict(clip=rectangle0, text="Project Alpha", font="times-roman", align=fitz.TEXT_ALIGN_LEFT)
Text_manipulator0 = PDFbf.manipulate_text(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), text=argument0.get("text"), font=argument0.get("font"), align=argument0.get("align"))
doc = Text_manipulator0.addtext()
# text string 2
argument1 = dict(clip=rectangle0, text="Phase 1", font="helv", rotate=0)
Text_manipulator1 = PDFbf.manipulate_text(doc=doc, pagenumber=pagenumber0, clip=argument1.get("clip"), text=argument1.get("text"), font=argument1.get("font"), rotate=argument1.get("rotate"), align=argument1.get("align"))
doc = Text_manipulator1.addtext()

# Page 5
pagenumber1=4
# rectangle 4
rectangleorder1=3
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)

# text string 3 base_level missing
missing_inf2 = '''file-level: a123gfsdfd_text.pdf, page-level: 5, order-level: 4, base-level: add text with font courier(incomplete)'''
recorder2 = PDFbf.recorder(missing_information=missing_inf2)
recorder2.recording

# text string 4
argument3 = dict(clip=rectangle1, text="Approved", textcolor="blue", fontsize=14, align=fitz.TEXT_ALIGN_JUSTIFY)
Text_manipulator3 = PDFbf.manipulate_text(doc=doc, pagenumber=pagenumber1, clip=argument3.get("clip"), text=argument3.get("text"), textcolor=argument3.get("textcolor"), fontsize=argument3.get("fontsize"), align=argument3.get("align"))
doc = Text_manipulator3.addtext()

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```
""",
    "6": """
Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks. 
Now, the drafter wants to edit the text content on the target area boxed by rectangle markups. Editing text on a target position is seen as a task. There are two possible operations must be select one: deleting and replacing. There are also format attributes that can be optionally specified on demand: the font, font size, alignment, text color, and rotation. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the necessary information of the operation) from high to low. The first three levels of information determine the target position, while the base level describes the operation. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. You need to find the four level information corresponding to each task from an instruction that may contain multiple tasks. Don't assume its value for necessary information if the command doesn't mention it or describe it vaguely. For all the tasks in commands, execute the tasks where all essential information is completely specified, while for tasks lacking necessary information, record the missing details. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.


Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for updating text content by modifying the argument's value specified by the drafter's instructions into a dictionary. 
5.	Update text content: Create a text manipulator instance for each text editing task, then extract the original text, revise the text content, and add new text to cover the original text.
6.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the text editing task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the text that needed to be revised;  "font": a string refers to the font of the text; "fontsize": an int refers to the font size; "align": a constant get by ```fitz.TEXT_ALIGN_RIGHT``` or ```fitz.TEXT_ALIGN_LEFT``` or ```fitz.TEXT_ALIGN_CENTER``` or ```fitz.TEXT_ALIGN_JUSTIFY``` refers to the align of the text in box; "rotate": a int in 0, or 90 refers to the direction of the original text to be edited, 0 means horizontal text, 90 means vertical text, if the direction of the text is not mentioned, the text is horizontal;}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"
Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"


Update text content:
A text manipulator is instantiated by specifying arguments (document(doc), page number(pagenumber), clip) and optional arguments (font, fontsize, text color, rotate, align) that mentioned like this: 
"Text_manipulator=PDFbf.manipulate_text(doc=fitz.document,pagenumber=int,clip=rectangle,font=string,fontsize=int,textcolor=string,rotate=int,align=fitz.TEXT_ALIGN_LEFT)"
You can extract the original horizontal text by calling the following function:
"Text_manipulator.text=Text_manipulator.gethortext()"
You can get a string of original vertical text by calling the following function:
"Text_manipulator.text=Text_manipulator.getvertext()"

Here are some arguments that should be specified when you want to delete or replace specified texts from the original texts:
"deltex": a string refers to the text the user wants to delete from the original text;
"retext": a string refers to the text the user wants to replace from the original text;
"totext": a string refers to the text the user wants to use to replace the original text;

You can delete specified text from the string of original text by defining the text to delete(deltex), and calling the following function:
"Text_manipulator.text = Text_manipulator.deletetext(deltex=string)"
"Text_manipulator.text = Text_manipulator.deletetext(deltex="target text")" delete the text "target text".
"Text_manipulator.text = Text_manipulator.deletetext()" delete all the text if a delete action is required with no deltext specified.

You can replace specified text with certain text for the string of original text by defining the text to replace(retext), text used to replace(totext) and calling the following function:
"Text_manipulator.text = Text_manipulator.replacetext(retext=string, totext=string)"
"Text_manipulator.text = Text_manipulator.replacetext(retext="old", totext="new")" replace the text "old" using text "new".
"Text_manipulator.text = Text_manipulator.replacetext(totext="1236")" replace all the text using text "1236" if a replace action is required with no retext specificed.
You can get a doc with original text covered by manipulated new text added by calling the following functions:
"doc = Text_manipulator.addtext()"

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" is a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no related part) or related part in the original instruction following with an "incomplete" in a brace (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an example of executing the user's instructions. You should respond in the same way to execute the user's new instructions.

User: The file name is "42Del9iutete_text.pdf". For the text in the second box of page 5, delete "temporary and text". Replace the text with "assignment". Use left align. Font size 8.
For the text in the fourth box of page 12, replace "manager" with "supervisor". Delete "completed" and "new".
For the text in page 18, replace "due" with "scheduled". Delete "soon". Font size 12.

Response:
```python

import fitz
import PDFbf

#Task 1
#File 1
filepath = "42Del9iutete_text.pdf"
doc = fitz.open(filepath)
annoextractor = PDFbf.extractanno(doc=doc)
#Page 1
pagenumber0 = 4
#Rectangle 1
rectangleorder0 = 1
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
#Operation 1
argument0 = dict(clip=rectangle0, align=fitz.TEXT_ALIGN_LEFT, fontsize=8)
Text_manipulator0 = PDFbf.manipulate_text(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), align=argument0.get("align"), fontsize=argument0.get("fontsize"))
Text_manipulator0.text = Text_manipulator0.gethortext()
Text_manipulator0.text = Text_manipulator0.deletetext(deltex="temporary and text")
Text_manipulator0.text = Text_manipulator0.replacetext(totext="assignment")
doc = Text_manipulator0.addtext()

#Task 2
#Page 2
pagenumber1 = 11
#Rectangle 2
rectangleorder1 = 3
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)
#Operation 2
argument1 = dict(clip=rectangle1)
Text_manipulator1 = PDFbf.manipulate_text(doc=doc, pagenumber=pagenumber1, clip=argument1.get("clip"))
Text_manipulator1.text = Text_manipulator1.gethortext()
Text_manipulator1.text = Text_manipulator1.replacetext(retext="manager", totext="supervisor")
Text_manipulator1.text = Text_manipulator1.deletetext(deltex="completed")
Text_manipulator1.text = Text_manipulator1.deletetext(deltex="new")
doc = Text_manipulator1.addtext()

#Task 3
#Lacking necessary information
missing_information2='''file-level:42Del9iutete_text.pdf,page-level:18,order-level:missing,base-level:replace "due" with "scheduled". Delete "soon". Font size 12.'''
recorder2 = PDFbf.recorder(missing_information=missing_information2)
recorder2.recording

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```
""",
    "7": """Text Mapping instruction.
Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks.
Now, the drafter wants to change the position of text on the target area boxed by rectangle markups. There are three operations that the drafter may want: translation, rotation, and scaling, the details defining the operation including operation type, direction and magnitude. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the details for at least one of the three operations) from high to low. What needs to be made more clear is that the parts of the details defining each type of operation that can be considered necessary information are pairs of information consisting of the type of operation and the direction, translation-direction, rotation-direction, and scale-factor or at least bigger or smaller than the original text, respectively. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks. Don't assume the value of missing necessary information (only the magnitude of operation can be assumed), but identify tasks lacking necessary information, record the missing details, and execute the task with all essential information gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for changing a text position by modifying the argument's value specified by the drafter's instructions into a dictionary. 
5.	Changing position: Create a text projector instance and change position.
6.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the text position-changing task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the text that need to be dealt with; "move": a list refers to translation details, general form: [move in its right direction (positive value if move right, negative value if move left), move in its upper direction (positive value if move up, negative value if move down), reference_point], "rotation": a list refers to the rotation details, general form: ['r', rotation degrees in the clockwise direction (positive value in clockwise direction, negative value in counterclockwise direction),reference_point], if the direction of rotation is not specified, clockwise is considered; "scal": a list refers to the scaling, general form: ['sc', [zoom factor X direction, zoom factor Y direction], reference_point]}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"
or get a list of all rectangles and their reference points:
"rectangles, rfpoints = annoextractor.getclip_rfpoint()"
The lists store all the rectangles and corresponding reference points detected. Showing their format with rectangles as an example : [[pagenumber,rectangleorder,rectangle]] 

Changing position:
A projector is instantiated by specifying the arguments (document(doc), page number(pagenumber), clip) and 1-3(at least one) optional arguments (translation (move), rotation(rotation) and scaling(scal)) on demand like this:
"text_projector=PDFbf.Projector(doc=fitz.document,pagenumber=int,clip=rectangle,move=[move_in_right_direction, move_in_up_direction, reference_point],rotation=['r', rotation degrees in clockwise direction,reference_point],scal=['sc',[zoom factor in X direction, zoom factor in Y direction], reference_point])"
You can get a doc with the text mapped by calling the following function:
"doc = text_projector.project()" 

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no related part) or related part in the original instruction following with an "incomplete" in brack (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an example. You should respond in the same way.

User: Move the text in the second box on the tenth page. For the file "47_mapqweqping_text.pdf", move the text in the first box on page 5 20 units left,30 units up. rotate the text 45 degrees clockwise. 
Response:
```python
import fitz
import PDFbf

# Task 1
# Recording the missing information for Task 1 (not enough details)
missing_information0 = '''file-level:incomplete, page-level: 10, order-level: 2, base-level: Move the text in the second box on the tenth page.(incomplete)'''
recorder0 = PDFbf.recorder(missing_information=missing_information0)
recorder0.recording()



# Task 2
# file 2 
filepath1 = "47_mapqweqping_text.pdf"
doc = fitz.open(filepath1)

annoextractor1 = PDFbf.extractanno(doc=doc)

# page 2
pagenumber1 = 4

# rectangle 2
rectangleorder1 = 0
rectangle1, rfpoint1 = annoextractor1.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)
argument1 = dict(clip=rectangle1, move=[-20, 30, rfpoint1], rotation=['r', 45, rfpoint1])
text_project1 = PDFbf.Projector(doc=doc, pagenumber=pagenumber1, clip=argument1.get("clip"), move=argument1.get("move"), rotation=argument1.get("rotation"))
doc = text_project1.project()

updatedfilepath = filepath1[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```

    """,
    "8": """
    Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks. 
Now, the drafter wants to update the text format on the target area boxed by rectangle markups. The possible format attributes are: font, font size, alignment, text color. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing at least one of the format attributes) from high to low. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks to complete the execution of the instruction. Don't assume the value of missing necessary information (if a format attribute is mentioned without specifying the value clearly, you can only assume that value), but identify tasks lacking necessary information, record the missing details, and execute the task only once all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for updating text format by modifying the argument's value specified by the drafter's instructions into a dictionary. 
5.	Update format: create a text manipulator instance, extract original text, refresh the text formate.
6.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the text position-changing task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the text that needed to be updated; "font": a string refers to the font of the text; "fontsize":an int refers to the font size; "align": a constant get by ```fitz.TEXT_ALIGN_RIGHT``` or ```fitz.TEXT_ALIGN_LEFT``` or ```fitz.TEXT_ALIGN_CENTER``` or ```fitz.TEXT_ALIGN_JUSTIFY``` refers to the align of the text in box;"rotate": a int in 0, or 90 refers to the direction of the text, 0 means horizontal text, 90 means vertical text, if the direction of the text is not mentioned, the text is horizontal text; "textcolor": a string refers to the color of the text}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"


Update format:
A text manipulator is instantiated by specifying arguments (document(doc), page number(pagenumber), clip) and optional arguments (font, fontsize, textcolor, rotate, align) on demand like this: "Text_manipulator=PDFbf.manipulate_text(doc=fitz.document,pagenumber=int,clip=rectangle,font=string,fontsize=int,textcolor=string,rotate=int, align=fitz.TEXT_ALIGN_LEFT)"
You can extract the original horizontal text by calling the following function:
"Text_manipulator.text = Text_manipulator.gethortext()"
You can get a string of original vertical text by calling the following function:
"Text_manipulator.text = Text_manipulator.getvertext()"
After extracting the original text, you can get a doc with the original text covered by refreshed new text added by calling the following functions:
"doc = Text_manipulator.addtext()"

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" is a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no related part) or related part in the original instruction following with an "incomplete" in a brace (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"


Here is an example. You should respond in the same way.
User: The file name is "234_Manipdfgwulate_text.pdf". For the vertical text in the second box of the fifth page, change the words color to green, center align. For the text in the fourth box of the eighth page, change the font to "times-roman" and the font size to 14. Update the text in the sixth box of the twelfth page.
Response:
```python

import fitz
import PDFbf

#task 1
#file 1
filepath = "234_Manipdfgwulate_text.pdf"
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc=doc)

#page 1
pagenumber0 = 4
#rectangle 1
rectangleorder0 = 1
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
argument0 = dict(clip=rectangle0, rotate=90, textcolor="green", align=fitz.TEXT_ALIGN_CENTER)
Text_manipulator0 = PDFbf.manipulate_text(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), rotate=argument0.get("rotate"), textcolor=argument0.get("textcolor"), align=argument0.get("align"))
Text_manipulator0.text = Text_manipulator0.getvertext()
doc = Text_manipulator0.addtext()

#task 2
#page 2
pagenumber1 = 7
#rectangle 2
rectangleorder1 = 3
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)
argument1 = dict(clip=rectangle1, font="times-roman", fontsize=14)
Text_manipulator1 = PDFbf.manipulate_text(doc=doc, pagenumber=pagenumber1, clip=argument1.get("clip"), font=argument1.get("font"), fontsize=argument1.get("fontsize"))
Text_manipulator1.text = Text_manipulator1.gethortext()
doc = Text_manipulator1.addtext()

#task 3
missing_information1 = '''file-level:234_Manipdfgwulate_text.pdf, page-level: 12, order-level: 6, base-level: missing'''
recorder2 = PDFbf.recorder(missing_information=missing_information1)

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```

    """,
    "9": """
Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks.   
Now, the drafter is trying to add the vector graphics drawn manually as standard drawings. The vector graphics have a parent class called "drawings" and three son classes, which are "rebar," "pillar/column," "line". Add a class of vector graphics is seen as a task.  The user will specify the target class he wants to deal with. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level(providing the annotation color) and base level(providing the vector class and format of new vectors) from high to low. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. What needs to be made more clear is that, the details defining a parent class that can be considered necessary information are: the color of annotation it is. You need to find the four level information corresponding to each task from an instruction that may contain multiple tasks. Don't assume its value for necessary information if the command doesn't mention it or describe it vaguely. For all the tasks in commands, execute the tasks where all essential information is completely specified, while for tasks lacking necessary information, record the missing details. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Argument collection: Modify the value of the argument specified by user instructions into dictionaries. Each dictionary contains the arguments for each stroke-adding action.
4.	Get target class of vector graphics
4.1	Parent Class Extraction: If a parent class("drawings") is asked to be added, you need to extract the parent class ("drawings") by creating an parent class extractor and then extract the parent class ("drawings"). 
4.2	Select Son Class: If a son class ("rebar", "pillar/column", "line", "rebars and columns") is asked to be added, you need to extract the parent class ("drawings") by creating an parent class extractor and then extract the parent class ("drawings") with the same "pagenumber" and "annocolor" as a prerequisite class. And then, create a son class selector, and select the target son classes ("rebar", "pillar/column", "line", "rebars and columns") from the prerequisite class.
5.	Strokes adding: Create a stroke manipulator with the target class of vector graphics and add them as standard strokes with the stroke manipulator for each action.
6.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the vector graphics adding task. Here is an example dictionary including all possible attributes:
{"annocolor": a string indicating the color of the annotation needed to be extracted; "drwcolor": a string indicating the color of the strokes when redrawing them as standard strokes; "width": an int indicating the line width of strokes when redrawing them as standard strokes}

The target class of vector graphics should named in this format: specified color + page number + class name: For example "C_4_rebars" means the rebar in color C on page 5. "A_3_columns" means the columns in color A on page 4. "T_0_lines" means the lines in color T on page 0.

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"
You have to import the library befor you using the following tools provided.

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Parent Class Extraction:
A parent class extractor can be instantiated by specifying the document(doc), page number(pagenumber), and annotation color(annocolor) like this:
"anno_extractor = PDFbf.extractanno (doc=fitz.document, pagenumber=int, annocolor=string)" 
You can extract a list of the parent class ("drawings") vector graphics in the annotation by calling the following function:
"color_pagenumber_drawings = anno_extractor.anno()";

Select Son Class:
A son classes selector can be instantiated by specifying the document(doc), page number(pagenumber), and the parent class with the same specified color and pagenumber of target son class(cdrawings) to extract a son classes ("rebars", "columns", "lines") with specified color and pagenumber, like this:
"color_pagenumber_son_classes_selector=PDFbf.select_from_drawings(doc=fitz.document, pagenumber=int, cdrawings=color_pagenumber_drawings)"
You can get a son classes ("rebars", "columns", "lines") with specified color on specified page from the parent class ("drawings") with the same specified color and page by calling the following functions:
"color_pagenumber_rebars=color_pagenumber_son_classes_selector.mode2_rebars_Cross_Touch_Intersect()", get a "rebars" son class with specified color on a specified page;
"color_pagenumber_columns=color_pagenumber_son_classes_selector.mode2_columns_Cross_Touch_Intersect()", get a "columns" son class with specified color on a specified page;
"color_pagenumber_lines=color_pagenumber_son_classes_selector.mode2_lines_Cross_Touch_Intersect()", get a "lines" son class with specified color on the specified page.
"color_pagenumber_rebar_and_column=color_pagenumber_son_classes_selector.get_rebar_column()", get a "rebars and columns" son class with specified color on the specified page.

Strokes adding:
A stoke manipulator can be instantiated by specifying the document(doc), page number(pagenumber), a list of the target class of vector graphics(sel_drawings) and optional arguments(stroke colors(drwcolor), line width of stroke(width)) like this:
"drawings_manipulator=PDFbf.manipulate_draw(doc=fitz.document, pagenumber=int, sel_drawings=list, drwcolor=string, width=int)"
You can get a doc with target vector graphics added by calling the following function:
"doc=drawings_manipulator.add_standrawing()"

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no realted part) or related part in the original instruction following with a "incompleted" in brack (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an example of executing the user's instructions. You should respond in the same way. 

User: The file name is "sdh.pdf". On the first page, add rebars marked by the red annotation, line width 3; Add drawings marked by blue annotation. The line color is green, line width is 2. On the third page, add columns. Line width is 4.
Response:
```python

import fitz
import PDFbf

#Task 1
#File 1
filepath = "sdh.pdf"
doc = fitz.open(filepath)
#Page 1
pagenumber0 = 0
#Target class 1
argument0 = dict(annocolor="red", width=3)
annoextractor0 = PDFbf.extractanno(doc=doc, pagenumber=pagenumber0, annocolor=argument0.get("annocolor"))
red_0_drawings = annoextractor0.anno()
red_0_son_classes_selector=PDFbf.select_from_drawings(doc=doc, pagenumber=pagenumber0, cdrawings=red_0_drawings)
red_0_rebars=red_0_son_classes_selector.mode2_rebars_Cross_Touch_Intersect() 
drawings_manipulator0 = PDFbf.manipulate_draw(doc=doc, pagenumber=pagenumber0, sel_drawings=red_0_rebars, width=argument0.get("width"))
doc = drawings_manipulator0.add_standrawing()

#Task 2
#Target class 2
argument1 = dict(annocolor="blue", drwcolor="green", width=2)
annoextractor1 = PDFbf.extractanno(doc=doc, pagenumber=pagenumber0, annocolor=argument1.get("annocolor"))
blue_0_drawings = annoextractor1.anno()
drawings_manipulator1 = PDFbf.manipulate_draw(doc=doc, pagenumber=pagenumber0, sel_drawings=blue_0_drawings, drwcolor=argument1.get("drwcolor"), width=argument1.get("width"))
doc = drawings_manipulator1.add_standrawing()

#Task 3
#Lacking necessary information
missing_information1='''file-level: sdh.pdf, page-level:3, order-level: missing, base-level: add columns. Line width is 4'''
recorder1 = PDFbf.recorder(missing_information=missing_information1)
recorder1.recording

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```
    """,
    "10": """
Task background:
You're a worker using tools provided by the drafter to implement his instruction on technical drawing editing tasks.  
Now, the drafter is trying to delete the vector graphics on the target area boxed by rectangle markups. The vector graphics have a parent class called "drawings" and three son classes, which are "rebar," "column," "line". There are two selection mode to select target class vectors, which are:  1. Window/Cover/Enclosure mode: the vector will be selected only if all vertices and boundaries of the graphic are covered within the rectangular checkbox, 2 Cross/Touch/Intersect mode: the vector will be selected if any of the vertices and a boundary of the vector are within the rectangular checkbox. Deleting vectors on a target position is seen as a task. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(the class(es) to be deleted) from high to low. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. What needs to be made more clear is that, the details defining a target class that can be considered necessary information are: the class and the selection mode. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks to complete the execution of the instruction. Don't assume the value of missing necessary information, but identify tasks lacking necessary information, record the missing details, and execute the task only once all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for deleting vectors on a target position by modifying the argument's value specified by the drafter's instructions into a dictionary. 
5.	Target class(es) selection: Create a drawing selector and select the specified vector graphics(drawings, lines, rebars, columns) with the specified selection mode by the selector.
6.	Delete vectors: Create a drawing manipulator or cleaner instance and delete the target vectors.
7.	Repair: Create a repairer instance if the repair action is asked for a target position, and repair vectors that should not be affected by the cleaner or manipulator; otherwise, skip this step.
8.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the vector deleting task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the drawings that needed to be dealt with}


Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"
You have to import the library befor you using the following tools provided.

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"


Target class(es) selection:
A drawing selector can be instantiated by specifying the document(doc), page number(pagenumber), and clip like this:
"selector = PDFbf.selector(doc=fitz.document, pagenumber=int, clip=rectangle)" 
The selector is to select a class with two optional selection modes. The class can be described by the type of vector graphics ("drawings", "lines", "rebars", or "columns"). There are two selection modes: 1. Window/Cover/Enclosure mode: the vector will be selected only if all vertices and boundaries of the graphic are covered within the rectangular checkbox, 2 Cross/Touch/Intersect mode: the vector will be selected if any of the vertices and a boundary of the vector are within the rectangular checkbox. If the selection mode is not mentioned, take selection mode 2 as default.
Choose the selection specified or closest to the description of the subgroup.
You can get a list of a subgroup with a specified selection mode by calling these functions:
"covered_drawings = selector.mode1_drawings_Window_Cover_Enclosure()" select drawings with selection mode 1;
"intersected_drawings = selector.mode2_drawings_Cross_Touch_Intersect()" select drawings with selection mode 2;
"covered_lines = selector.mode1_lines_Window_Cover_Enclosure()" select lines with selection mode 1;
"intersected_lines = selector.mode2_lines_Cross_Touch_Intersect()" select lines with selection mode 2;
"covered_rebars= selector.mode1_rebars_Window_Cover_Enclosure()" select rebars with selection mode 1;
"intersected_rebars = selector.mode2_rebars_Cross_Touch_Intersect()" select rebars with selection mode 2;
"covered_columns = selector.mode1_columns_Window_Cover_Enclosure()" select columns with selection mode 1;
"intersected_columns = selector.mode2_columns_Cross_Touch_Intersect()" select columns with selection mode 2;


Delete vectors:
1. Delete selected class:
A drawing manipulator is instantiated by specfying the document(doc), page number(pagenumber), and list of drawings(listofcdraw) like this:
"drawing_manipulator = PDFbf.draw_drawer(doc= fitz.document, pagenumber=int, listofcdraw=list)"
You can get a doc with the list of drawings deleted by calling following functions:
"doc = drawing_manipulator.delete_draw()"

2. Delete indiscriminately:
An indiscriminate cleaner is instantiated by specifying the document(doc), page number(pagenumber), and clip like this:
"cleaner=PDFbf.delete(doc=fitz.document, pagenumber=int, clip=rectangle)"
You can get a doc with everything in the clip covered by calling the following function:
"doc = cleaner.applydelete()"

Repair:
A repairer is instantiated by specifying the document(doc), page number(pagenumber), clip, cdrawings(cdrawings) and optional specify selected drawing(drawings) like this:
"repairer=PDFbf.repairer(doc= fitz.document,pagenumber=int,clip=rectangle, cdrawings=list, sel_drawings=list)"
"cdrawings" can be obtained by calling the selector like this "selector.selected_lines", "sel_drawings" can be specified as None to repair the affect by an indiscriminate cleaner.
You can get a lines repaired doc by calling this function:
"doc = repairer.del_repair()"
Selected drawings should be the total selected drawings.

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no realted part) or related part in the original instruction following with a "incompleted" in brack (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording()"

Here is an example. You should respond in the same way.

User: The file name is "sdh.pdf". For the first box on page three, delete the drawings covered by the box, delete the rebars intersected with the box, and delete the columns covered by the box. For the fourth box on page eleven, delete the lines intersecting with the box. For the eighth box on page one, delete the lines covered by the box and delete the rebars covered by the box. For the second box on page 18, delete. Repair all the drawings.

Response:
```python

import fitz
import PDFbf

filepath = "sdh.pdf"
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc=doc)


# Mission 1
pagenumber0 = 2
rectangleorder0 = 0

rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
argument0 = dict(clip=rectangle0)
selector0 = PDFbf.selector(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"))
covered_drawings0 = selector0.mode1_drawings_Window_Cover_Enclosure()
intersected_rebars0 = selector0.mode2_rebars_Cross_Touch_Intersect()
covered_columns0 = selector0.mode1_columns_Window_Cover_Enclosure()
sel_drawings0 = covered_drawings0 + intersected_rebars0 + covered_columns0
drawing_manipulator0 = PDFbf.draw_drawer(doc=doc, pagenumber=pagenumber0, listofcdraw=sel_drawings0)
doc = drawing_manipulator0.delete_draw()
repairer0 = PDFbf.repairer(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), cdrawings=selector0.selected_lines, sel_drawings=sel_drawings0)
doc = repairer0.del_repair()

# Mission 2
pagenumber1 = 10
rectangleorder1 = 3
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)
argument1 = dict(clip=rectangle1)
selector1 = PDFbf.selector(doc=doc, pagenumber=pagenumber1, clip=argument1.get("clip"))
intersected_lines1 = selector1.mode2_lines_Cross_Touch_Intersect()
drawing_manipulator1 = PDFbf.draw_drawer(doc=doc, pagenumber=pagenumber1, listofcdraw=intersected_lines1)
doc = drawing_manipulator1.delete_draw()
repairer1 = PDFbf.repairer(doc=doc, pagenumber=pagenumber1, clip=argument1.get("clip"), cdrawings=selector1.selected_lines, sel_drawings=intersected_lines1)
doc = repairer1.del_repair()

# Mission 3
pagenumber2=0
rectangleorder2=7
rectangle2, rfpoint2 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder2)
argument2 = dict(clip=rectangle2)
selector2=PDFbf.selector(doc=doc,pagenumber=pagenumber2,clip=argument2.get("clip"))
covered_lines2=selector2.mode1_lines_Window_Cover_Enclosure()
covered_rebars2 = selector2.mode1_rebars_Window_Cover_Enclosure()
sel_drawings2 = covered_lines2 + covered_rebars2
drawing_manipulator2 = PDFbf.draw_drawer(doc=doc,pagenumber=pagenumber2,listofcdraw=sel_drawings2)
doc = drawing_manipulator2.delete_draw()
repairer2 = PDFbf.repairer(doc=doc, pagenumber=pagenumber2, clip=argument2.get("clip"), cdrawings=selector2.selected_lines, sel_drawings=sel_drawings2)
doc = repairer2.del_repair()

# Mission 4
missing_information3='''file-level: sdh.pdf, page-level:18, order-level:2, base-level: delete(incomplete)'''
recorder3 = PDFbf.recorder(missing_information=missing_information3)
recorder3.recording

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```
    """,
    "11": """
Task background:
You're a high-level programmer. You are writing code to help a drafter implement construction drawings editing tasks. 
Now, the drafter is trying to map the vector graphics on the target area boxed by rectangle markups. The vector graphics have a parent class called "drawings" and three son classes, which are "rebar," "column," "line". There are two selection mode to select target class vectors, which are:  1. Window/Cover/Enclosure mode: the vector will be selected only if all vertices and boundaries of the graphic are covered within the rectangular checkbox, 2 Cross/Touch/Intersect mode: the vector will be selected if any of the vertices and a boundary of the vector are within the rectangular checkbox. There are three operations that the drafter may want: translation, rotation, and scaling, the details defining the operation including operation type, direction and magnitude. Deleting a group of vectors on a target position with a corresponding group of operations is seen as a task. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the classes of vectors to be mapped and providing the details for at least one of the three operations correspondingly) from high to low. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. What needs to be made more clear is that, the details defining a target class that can be considered necessary information are: the class and the selection mode. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks to complete the execution of the instruction. Don't assume the value of missing necessary information (only the magnitude of operation can be assumed), but identify tasks lacking necessary information, record the missing details, and execute the task only once all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for mapping vectors on a target position by modifying the argument's value specified by the drafter's instructions into a dictionary. 
5.	Target class(es) selection: Create a drawing selector and select the specified vector graphics(drawings, lines, rebars, columns) with the specified selection mode by the selector.
6.	Mapping vectors: Create a vectors projector instance and map vectors.
7.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;


Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the table position-changing task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the text that need to be dealt with; "move": a list refers to translation details, general form: [move in its right direction (positive value if move right, negative value if move left), move in its upper direction (positive value if move up, negative value if move down), reference_point], "rotation": a list refers to the rotation details, general form: ['r', rotation degrees in the clockwise direction (positive value in clockwise direction, negative value in counterclockwise direction),reference_point], if the direction of rotation is not specified, clockwise is considered; "scal": a list refers to the scaling, general form: ['sc', [zoom factor X direction, zoom factor Y direction], reference_point]}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"
You have to import the library befor you using the following tools provided.

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"
or get a list of all rectangles and their reference points:
"rectangles, rfpoints = annoextractor.getclip_rfpoint()"
The lists store all the rectangles and corresponding reference points detected. Showing their format with rectangles as an example : [[pagenumber,rectangleorder,rectangle]] 

Target class(es) selection:
A vectors selector can be instantiated by specifying the document(doc), page number(pagenumber), and clip like this:
"selector = PDFbf.selector(doc=fitz.document, pagenumber=int, clip=rectangle)" 
The selector is to select a subgroup from drawings with two optional selection modes. The subgroup can be described by the type of vector graphics ("drawings", "lines", "rebars", or "columns"). There are two selection modes: 1. Window/Cover/Enclosure mode: the vector will be selected only if all vertices and boundaries of the graphic are completely within the rectangular checkbox, 2 Cross/Touch/Intersect mode: the vector will be selected if any of the vertices and a boundary of the vector are within the rectangular checkbox.
Choose the selection specified or closest to the description of the subgroup.
You can get a list of a subgroup with a specified selection mode by calling these functions:
"covered_drawings = selector.mode1_drawings_Window_Cover_Enclosure()" select drawings with selection mode 1;
"intersected_drawings = selector.mode2_drawings_Cross_Touch_Intersect()" select drawings with selection mode 2;
"covered_lines = selector.mode1_lines_Window_Cover-Enclosure()" select lines with selection mode 1;
"intersected_lines = selector.mode2_lines_Cross_Touch_Intersect()" select lines with selection mode 2;
"covered_rebars= selector.mode1_rebars_Window_Cover_Enclosure()" select rebars with selection mode 1;
"intersected_rebars = selector.mode2_rebars_Cross_Touch_Intersect()" select rebars with selection mode 2;
"covered_columns = selector.mode1_columns_Window_Cover_Enclosure()" select columns with selection mode 1;
"intersected_columns = selector.mode2_columns_Cross_Touch_Intersect()" select columns with selection mode 2;

Mapping vectors:
A vector projector is instantiated by specifying the essential arguments (document(doc), page number(pagenumber), clip, selected drawings(sel_drawings),cdrawings(cdrawings)) and optional arguments (movement of translation (move), rotation(rotation) and scaling(scal)) on demand like this:
"projector=PDFbf.project_draw(doc=fitz.document,pagenumber=int,clip=rectangle,sel_drawings=list, cdrawings=list, move=[move_in_right_direction, move_in_up_direction, reference_point],rotation=['r', rotation degrees in clockwise direction,reference_point],scal=['sc',[zoom factor X direction, zoom factor Y direction], reference_point])"
You can get a doc with the drawings mapped by calling the following function:
"doc = projector.project()"
"cdrawings" can be obtained by calling the selector like this "selector.selected_lines",

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no realted part) or related part in the original instruction following with a "incompleted" in brack (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an example. You should respond in the same way.

User: The file name is "fjhgdf.pdf". For the eighth box on page one, rotate the drawing covered by the box 45 degrees clockwise; move the rebars intersected with the box 50 units left, 20 units down. For the second box on page five, scale the columns intersected with the box 0.8 times in both x and y directions; rotate the rebars covered by the box 45 degrees counterclockwise and move it 30 units left, 20 units up and scal it 0.8 times in x direction, 1.2 times in y firection. For the first box on page seven, scale the rebars covered by the box 1.5 times in both x and y directions; move the lines covered by the box 20 units right, 10 units up, and rotate it. 

Response:
```python

import fitz
import PDFbf

filepath = "fjhgdf.pdf"
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc=doc)

pagenumber0=0
rectangleorder0=7
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0,rectangleorder=rectangleorder0)
argument0=dict(clip=rectangle0, rotation=['r', 45, rfpoint0])
selector0 = PDFbf.selector(doc=doc,pagenumber=pagenumber0,clip=argument0.get("clip"))
covered_drawings0 = selector0.mode1_drawings_Window_Cover_Enclosure()
projector0 = PDFbf.project_draw(doc=doc, pagenumber=pagenumber0, clip=argument0.get("clip"), sel_drawings=covered_drawings0, cdrawings=selector0.selected_lines, move=argument0.get("rotation"))
doc = projector0.project()

argument1=dict(clip=rectangle0, move=[-50,-20, rfpoint0])
intersected_rebars1 = selector0.mode2_rebars_Cross_Touch_Intersect()
projector1 = PDFbf.project_draw(doc=doc, pagenumber=pagenumber0, clip=argument1.get("clip"), sel_drawings=intersected_rebars1, cdrawings=selector0.selected_lines, rotation=argument1.get("move")）
doc = projector1.project()


pagenumber1 = 4
rectangleorder1 = 1
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1,rectangleorder=rectangleorder1)
argument2 = dict(clip=rectangle1, scal=['sc', [0.8, 0.8], rfpoint1])
selector1 = PDFbf.selector(doc=doc, pagenumber=pagenumber1, clip=argument2.get("clip"))
intersected_columns2 = selector1.mode2_columns_Cross_Touch_Intersect()
projector2 = PDFbf.project_draw(doc=doc, pagenumber=pagenumber1, clip=argument2.get("clip"), sel_drawings=intersected_columns2, cdrawings=selector1.selected_lines, scal=argument2.get("scal"))
doc = projector2.project()

argument3 = dict(clip=rectangle1, move=[-30, 20, rfpoint1], rotation=['r', -45, rfpoint1], scal=['sc', [0.8, 1.2], rfpoint1])
covered_rebars3 = selector1.mode1_rebars_Window_Cover_Enclosure()
projector3 = PDFbf.project_draw(doc=doc, pagenumber=pagenumber1, clip=argument3.get("clip"), sel_drawings=covered_rebars3, cdrawings=selector1.selected_lines, rotation=argument3.get("rotation"), move=argument3.get("move"), scal=argument3.get("scal"))
doc = projector3.project()

pagenumber2 = 6
rectangleorder2 = 0
rectangle2, rfpoint2 = annoextractor.getclip_rfpoint(pagenumber=pagenumber2,rectangleorder=rectangleorder2)
argument4 = dict(clip=rectangle2, scal=['sc', [1.5, 1.5], rfpoint2])
selector2 = PDFbf.selector(doc=doc, pagenumber=pagenumber2, clip=argument4.get("clip"))
covered_rebars4 = selector2.mode1_rebars_Window_Cover_Enclosure()
projector4 = PDFbf.project_draw(doc=doc, pagenumber=pagenumber2, clip=argument4.get("clip"), sel_drawings=covered_rebars4, cdrawings=selector2.selected_lines, scal=argument4.get("scal"))
doc = projector4.project()

missing_information5='''file-level: fjhgdf.pdf, page-level:7, order-level:1, base-level: move the lines covered by the box 20 units right, 10 units up, and rotate it (incomplete)'''
recorder5 = PDFbf.recorder(missing_information=missing_information5)
recorder5.recording

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```    
    """,
    "12": """
 Task background:
You're a high-level programmer. You are writing code to help a drafter implement construction drawings editing tasks.  
Now, the drafter is trying to update the vector graphics format on the target area boxed by rectangle markups. The vector graphics have a parent class called "drawings" and three son classes, which are "rebar," "column," "line". There are two selection modes to select target class vectors, which are:  1. Window/Cover/Enclosure mode: the vector will be selected only if all vertices and boundaries of the graphic are covered within the rectangular checkbox, 2 Cross/Touch/Intersect mode: the vector will be selected if any of the vertices and a boundary of the vector are within the rectangular checkbox. There are all possible format attributes that the drafter may want to change: stroke color, line type (dash or straight), close path, fill color, joint type, cap type, and stroke width. Updating a group of vectors on a target position with a corresponding group of format attributes is seen as a task. The necessary information for such a task has four levels, which are file-level (providing the file path), page-level (providing page number), order-level (providing rectangle order on the page), and base-level(providing the classes of vectors to be updated and providing the details for at least one of the format attributes correspondingly) from high to low. A lower level information can correspond to only one higher level information, while higher level information can correspond to multiple lower-level information. What needs to be clarified is that the details defining a target class that can be considered necessary information are the class and the selection mode. You need to find the necessary information corresponding to each task from an instruction that may contain multiple tasks to complete the execution of the instruction. Don't assume the value of missing necessary information (only the value of the format attribute mentioned but not specified value explicitly can be assumed), but identify tasks lacking necessary information, record the missing details, and execute the task only once all essential information is gathered. Don't repeat the operations related to the same higher-level information when dealing with lower-level operations. Avoid using loops.

Task plan:
The following is a standard process for completing a task:
0.	Inexecutable task recording: If you think a level of necessary information(file-level, page-level, order-level, base-level) for a task is missing in the instruction, don't assume the value but skip all the operations of that task and create a recorder to record the incomplete instruction, then continue on next task. Otherwise, ignore this step if all four levels of necessary information of a task can be found.
1.	Import libraries
2.	Open the file.
3.	Target position extraction: Crate an annotation extractor and extract the rectangles on the target page and order.
4.	Argument collection: Prepare parameters for updating vectors on a target position by modifying the argument's value specified by the drafter's instructions into a dictionary.  
5.	Target class(es) selection: Create a drawing selector and select the specified vector graphics(drawings, lines, rebars, columns) with the specified selection mode by the selector.
6.	Updating vectors: Create a vector manipulating instance and update vectors.
7.	Save the file.

Available tools:
Two constant defined by the user:
"pagenumber": an int refers to the page number, pagenumber=0 indicates page one; 
"rectangleorder" an int refers to the order of rectangles within the page, rectangleorder=0 indicates the first rectangle;

Argument collection:
Extract the argument values from the instruction and store them as a dictionary for the vector updating task. Here is an example dictionary including all possible attributes:
{"clip": a rectangle boxes the drawings that needed to be dealt with; "fillcolor": a string refers to the fill color; "drwcolor": a string refers to the color of the drawing; "dashes": a sting refers to the dashed line, the general format is "[n m] 0", n is an int indicting the dash length, m (optional) is an int indicting the subsequent gap length, and 0 is an int specifying no pixels should be skipped before the dashing starts, if a dashed line is required, but no parameters are specified, take "[3 3]0", if a straight line is required, the parameters should be "[0 0]0"; "closePath": a bool indicating whether the path is a close path; "lineJoin": an int controlling the way how line connections look like. This may be either as a sharp join (0), a rounded join (1), or a cut-off join(2);" lineCap": an int controlling the look of line ends. the options are sharp ends(0), semi-circle ends(1), and semi-square ends(2); "width": an int refers to the line width of drawings}

Popular PDF editing library "PyMuPDF" and a user-tailored library "PDFbf" are provided. You can import the two libraries by:
"import fitz
import PDFbf"
You have to import the library befor you using the following tools provided.

File open:
You can open the file by calling the following function:
"doc = fitz.open("filepath")" Open the file with the file path.

Save the updated file:
You can save the file with an updated name by calling the following function:
"
updated_file = filepath[:-4] + "_updated.pdf" # update file path
doc.save(updated_file)
"

Target position extraction:
An annotation extractor is instantiated by providing the essential arguments document(doc):
"annoextractor = PDFbf.extractanno(doc=fitz.document)"
Get an rectangle and its reference pointsGet a list of all rectangles and their reference points:
"rectangle, rfpoint = annoextractor.getclip_rfpoint(pagenumber=int,rectangleorder=int)"


Target class(es) selection:
A vectors selector can be instantiated by specifying the document(doc), page number(pagenumber), and clip like this:
"selector = PDFbf.select_from_drawings(doc=fitz.document, pagenumber=int, clip=rectangle)" 
The selector is to select a subgroup from drawings with two optional selection modes. The subgroup can be described by the type of vector graphics ("drawings", "lines", "rebars", or "columns"). There are two selection modes: 1. Window/Cover/Enclosure mode: the vector will be selected only if all vertices and boundaries of the graphic are completely within the rectangular checkbox, 2 Cross/Touch/Intersect mode: the vector will be selected if any of the vertices and a boundary of the vector are within the rectangular checkbox.
Choose the selection specified or closest to the description of the subgroup.
You can get a list of a subgroup with a specified selection mode by calling these functions:
"covered_drawings = selector.mode1_drawings_Window_Cover_Enclosure()" select drawings with selection mode 1;
"intersected_drawings = selector.mode2_drawings_Cross_Touch_Intersect()" select drawings with selection mode 2;
"covered_lines = selector.mode1_lines_Window_Cover_Enclosure()" select lines with selection mode 1;
"intersected_lines = selector.mode2_lines_Cross_Touch_Intersect()" select lines with selection mode 2;
"covered_rebars= selector.mode1_rebars_Window_Cover_Enclosure()" select rebars with selection mode 1;
"intersected_rebars = selector.mode2_rebars_Cross_Touch_Intersect()" select rebars with selection mode 2;
"covered_columns = selector.mode1_columns_Window_Cover_Enclosure()" select columns with selection mode 1;
"intersected_columns = selector.mode2_columns_Cross_Touch_Intersect()" select columns with selection mode 2;


Updating vectors:
A vector manipulator is instantiated by specifying the essential arguments (document(doc), page number(pagenumber), selected drawings(sel_drawings)) and optional arguments (fill color (fillcolor), drawing color(drwcolor), dashes line (dashes), close path or not (closePath), line joint shape (lineJoin), line end shape (lineCap), line width (width)) on demand like this:
"manipulator=PDFbf.manipulate_draw(doc=doc,pagenumber=int,sel_drawings=list,fillcolor=string,drwcolor=string,dashes=string,closePath=bool,lineJoin=int,lineCap=int,width=int)"
You can get a doc with the list of drawings refreshed by calling the following function:
"doc = manipulator.update_draw()"

Incomplete instruction recording: 
A missing key attribute recorder can be instantiated to record the incomplete instruction by providing the essential arguments(incomplete_instruction, missing_information):
"recorder = PDFbf.recorder(missing_information=str)"
Where "missing_information" a string indicating the highest-level of missing information, a standard format is '''file-level: value or missing, page-level: value or missing, order-level: value or missing, base-level: related part in the original instruction or missing(if no realted part) or related part in the original instruction following with a "incompleted" in brack (if there is some related description but some necessary information is missing) '''.
The incomplete instruction can be recorded by calling the following function:
"recorder.recording"

Here is an example. You should respond in the same way.

User: The file name is "fjhgdf.pdf". For the third box on page two, change the columns intersecting with the box to red, dash lines with rounded line joints; change the line covered by the box to a dashed line and line width 3. For the eighth box on page one, change the lines intersecting with the box to yellow filling, close path, semi-square line end; change the rebars covered by the box to blue with a dashed line and line width 3. For the fifth box on page ten, change the columns covered by the box to green with a semi-square line end and line width 4; change the rebars intersecting with the box.

Response:
```python

import fitz
import PDFbf
filepath = "fjhgdf.pdf"
doc = fitz.open(filepath)

annoextractor = PDFbf.extractanno(doc)

# Mission 1
pagenumber0=1
rectangleorder0=2
rectangle0, rfpoint0 = annoextractor.getclip_rfpoint(pagenumber=pagenumber0, rectangleorder=rectangleorder0)
argument0=dict(clip=rectangle0,drwcolor="red",dashes="[3,3]0",lineJoin=1)
selector0 = PDFbf.selector(doc=doc,pagenumber=pagenumber0,clip=argument0.get("clip"))
intersected_columns0 = selector0.mode2_columns_Cross_Touch_Intersect()
manipulator0=PDFbf.manipulate_draw(doc=doc,pagenumber=pagenumber0,sel_drawings=intersected_columns0,drwcolor=argument0.get("drwcolor"),dashes=argument0.get("dashes"),lineJoin=argument0.get("lineJoin"))
doc = manipulator0.update_draw()

argument1=dict(clip=rectangle0,dashes="[3,3]0",width=3)
covered_lines1 = selector0.mode1_lines_Window_Cover_Enclosure()
manipulator1=PDFbf.manipulate_draw(doc=doc,pagenumber=pagenumber0,sel_drawings=covered_lines1,dashes=argument1.get("dashes"),width=argument1.get("width"))
doc = manipulator1.update_draw()

pagenumber1=0
rectangleorder1=7
rectangle1, rfpoint1 = annoextractor.getclip_rfpoint(pagenumber=pagenumber1, rectangleorder=rectangleorder1)
argument2=dict(clip=rectangle1,fillcolor="yellow",closePath=True,lineCap=2)
selector1 = PDFbf.selector(doc=doc,pagenumber=pagenumber1,clip=argument2.get("clip"))
intersected_lines2 = selector1.mode2_lines_Cross_Touch_Intersect()
manipulator2=PDFbf.manipulate_draw(doc=doc, pagenumber=pagenumber1, sel_drawings=intersected_lines2, fillcolor =argument2.get("fillcolor"), closePath=argument2.get("closePath"), lineCap=argument2.get("lineCap"))
doc = manipulator2.update_draw()

argument3 = dict(clip=rectangle1, drwcolor="blue", dashes="[3 3]0", width=3)
covered_rebars3 = selector1.mode1_rebars_Window_Cover_Enclosure()
manipulator3 = PDFbf.manipulate_draw(doc=doc, pagenumber=pagenumber1, sel_drawings=covered_rebars3, drwcolor=argument3.get("drwcolor"), dashes=argument3.get("dashes"), width=argument3.get("width"))
doc = manipulator3.update_draw()

# Mission 3
pagenumber2 = 9
rectangleorder2 = 4
rectangle2, rfpoint2 = annoextractor.getclip_rfpoint(pagenumber=pagenumber2, rectangleorder=rectangleorder2)
argument4 = dict(clip=rectangle2, drwcolor="green", lineCap=2, width=4)
selector2 = PDFbf.selector(doc=doc, pagenumber=pagenumber2, clip=argument4.get("clip"))
covered_columns4 = selector2.mode1_columns_Window_Cover_Enclosure()
manipulator4 = PDFbf.manipulate_draw(doc=doc, pagenumber=pagenumber2, sel_drawings=covered_columns4, drwcolor=argument4.get("drwcolor"), lineCap=argument4.get("lineCap"), width=argument4.get("width"))
doc = manipulator4.update_draw()

missing_information5='''file-level: fjhgdf.pdf, page-level:10, order-level:5, base-level: change the rebars intersecting with the box (incomplete)'''
recorder5 = PDFbf.recorder(missing_information=missing_information5)
recorder5.recording

updatedfilepath = filepath[:-4] + "_updated.pdf"
doc.save(updatedfilepath)
```

    """,
}
