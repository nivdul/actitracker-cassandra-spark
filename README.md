# actitracker-cassandra-spark

The availability of acceleration sensors creates exciting new opportunities for data mining and predictive analytics applications. In this post, I will consider data from accelerometers to perform activity recognition.

## Data description
We use labeled accelerometer data from users thanks to a device in their pocket during different activities 
(walking, sitting, jogging, ascending stairs, descending stairs, and standing).

The accelerometer measures acceleration in all three spatial dimensions as following:

- Z-axis captures the forward movement of the leg
- Y-axis captures the upward and downward movement of the leg
- X-axis captures the horizontal movement of the leg

The plots below show characteristics for each activity. Because of the periodicity of such activities, a few seconds windows is sufficient.


  <div>
		<a href="/img/accelerometer_walking.jpg" ><img src="/img/grouped_images.jpg"  alt="Walking activity"></a>
  	</div>


We observe repeating waves and peaks for the following repetitive activities walking, jogging, ascending stairs and descending stairs. The activities Upstairs and Downstairs are very similar, and there is no periodic behavior for more static activities like standing or sitting, but different amplitudes.

More about on this [post]().

##Launch the project

###Cassandra
I have pushed the data into Cassandra and then used the [connector Spark-Cassandra](https://github.com/datastax/spark-cassandra-connector).

To create a table and push your data:
<script src="https://gist.github.com/nivdul/88d1dbb944f75c8bf612.js"></script>

###Data

The data as a cvs file are available in the directory '/data/data.csv.zip'.
You can also find the features in the zip '/data/features-actitracker.zip'.
