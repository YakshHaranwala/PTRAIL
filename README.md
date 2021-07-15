
<!---------------------- Introduction Section ------------------->
<h1> NumMobility: A Mobility Data PreProcessing Library </h1>

<h2> Introduction </h2>

<p align='justify'>
NumMobility is a state-of-the art Mobility Data Preprocessing Library that mainly deals with filtering data, generating features and interpolation of Trajectory Data.

<b><i> The main features of NumMobility are: </i></b>
</p>

<ol align='justify'>
<li> NumMobility uses primarily parallel computation based on
     python Pandas and numpy which makes it very fast as compared
     to other libraries available.
</li>

<li> NumMobility harnesses the full power of the machine that
     it is running on by using all the cores available in the
     computer.
</li>

<li> NumMobility uses a customized DataFrame built on top of python
     pandas for representation and storage of Trajectory Data.
</li>

<li> NumMobility also provides several Temporal and spatial features
     which are calculated mostly using parallel computation for very
     fast and accurate calculations.
</li>

<li> Moreover, NumMobility also provides several filteration and
     outlier detection methods for cleaning and noise reduction of
     the Trajectory Data.
</li>

<li> Apart from the features mentioned above, <i><b> four </b></i>
     different kinds of Trajectory Interpolation techniques are
     offered by NumMobility which is a first in the community.
</li>
</ol>

<!----------------- Dataset Link Section --------------------->
<hr style="height:6px;background-color:black">

<p align='justify'>
In the introduction of the library, the seagulls dataset is used
which can be downloaded from the link below: <br>
<span> &#8618; </span>
<a href="https://github.com/YakshHaranwala/NumMobility/blob/main/examples/data/gulls.csv" target='_blank'> Seagulls Dataset </a>
</p>

<!----------------- NbViewer Link ---------------------------->
<hr style="height:6px;background-color:black">
<p align='justify'>
Note: Viewing this notebook in GitHub will not render JavaScript
elements. Hence, for a better experience, click the link below
to open the Jupyter notebook in NB viewer.

<span> &#8618; </span>
<a href="https://nbviewer.jupyter.org/github/YakshHaranwala/NumMobility/blob/main/examples/0.%20Intro%20to%20NumMobility.ipynb" target='_blank'> Click Here </a>
</p>

<!------------------------- Documentation Link ----------------->
<hr style="height:6px;background-color:black">
<p align='justify'>
The Link to NumMobility's Documentation is: <br>
</p>

<span> &#8618; </span>
<a href='https://nummobility.readthedocs.io/en/latest/' target='_blank'> <i> NumMobility Documentation </i> </a>
<hr style="height:6px;background-color:black">
<h2> Importing Trajectory Data into a NumPandasTraj Dataframe </h2>

<p align='justify'>
NumMobility Library stores Mobility Data (Trajectories) in a specialised
pandas Dataframe structure called NumPandasTraj. As a result, the following
constraints are enforced for the data to be able to be stores in a NumPandasTraj.
</p>

<ol align='justify'>
   <li>
        Firstly, for a mobility dataset to be able to work with NumMobility Library needs
        to have the following mandatory columns present:
       <ul type='square'>
           <li> DateTime </li>
           <li> Trajectory ID </li>
           <li> Latitude </li>
           <li> Longitude </li>
       </ul>
   </li>
   <li>
       Secondly, NumPandasTraj has a very specific constraint for the index of the
       dataframes, the Library enforces a multi-index consisting of the
       <b><i> Trajectory ID, DateTime </i></b> columns because the operations of the
       library are dependent on the 2 columns. As a result, it is recommended
       to not change the index and keep the multi-index of <b><i> Trajectory ID, DateTime </i></b>
       at all times.
   </li>
   <li>
        Note that since NumPandasTraj Dataframe is built on top of
        python pandas, it does not have any restrictions on the number
        of columns that the dataset has. The only requirement is that
        the dataset should atleast contain the above mentioned four columns.
   </li>
</ol>

<hr style="height:6px;background-color:black">