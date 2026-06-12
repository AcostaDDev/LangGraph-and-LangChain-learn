from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv


load_dotenv()
model1 = ChatGroq(model="llama-3.3-70b-versatile")
model2 = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate short and simple note from the following text: {text}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following text: {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and questions-answers into a single socument. Notes: {notes}. Questions-Answers: {qa}",
    input_variables=['notes', 'qa']
)

runnable_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'qa': prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser
final_chain = runnable_chain | merge_chain

text = """
A Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used primarily for data classification, but it can also be applied to regression tasks and anomaly detection. The core objective of an SVM is to find the optimal decision boundary—known as a hyperplane—that separates different classes of data by the widest possible margin.Core Concepts of SVMTo understand how an SVM works, it helps to break it down into its foundational components:Hyperplane: This is the decision boundary that separates the data classes. In a 2D space (two features), a hyperplane is simply a straight line. In a 3D space, it is a flat plane. For higher dimensions, it is an N-dimensional geometric surface.Support Vectors: These are the specific data points that lie closest to the hyperplane. They are critical because they anchor the decision boundary; if you remove these points, the position of the hyperplane shifts.Margin: This is the perpendicular distance between the hyperplane and the closest support vectors. SVM is a "max-margin" model, meaning it explicitly searches for the boundary that creates the largest possible gap (or "street") between classes to maximize accuracy on new data.Linear vs. Non-Linear SVMData in the real world is rarely perfectly organized, which splits SVM implementations into two approaches:1. Linear SVMUsed when data can be perfectly separated by a straight line or plane. The algorithm draws a clean boundary to isolate the categories without altering the original coordinates.2. Non-Linear SVM & The Kernel TrickWhen data points are tangled together and cannot be separated linearly, SVM uses a mathematical workaround called the Kernel Trick.Instead of trying to bend a complex curve around the data in its original format, the kernel function projects the data into a higher-dimensional space.In this new, higher dimension, the data becomes spread out enough that a flat, linear hyperplane can cleanly cut between the classes.Popular kernel types include Linear, Polynomial, and Radial Basis Function (RBF).
"""

result = final_chain.invoke(text)
print(result)