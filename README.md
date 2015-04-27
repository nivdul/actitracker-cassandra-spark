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

We observe repeating waves and peaks for the following repetitive activities walking, jogging, ascending stairs and descending stairs. The activities Upstairs and Downstairs are very similar, and there is no periodic behavior for more static activities like standing or sitting, but different amplitudes.

More about on this [post](https://nivdul.wordpress.com/2015/04/20/analyze-accelerometer-data-with-apache-spark-and-mllib/).

##Launch the project

###Cassandra
I have pushed the data into Cassandra and then used the [connector Spark-Cassandra](https://github.com/datastax/spark-cassandra-connector).

To create a table and push your data:
```sql
CREATE KEYSPACE actitracker WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1 };
CREATE TABLE users (user_id int,activity text,timestamp bigint,acc_x double,acc_y double,acc_z double, PRIMARY KEY ((user_id,activity),timestamp));
COPY users FROM '/path_to_your_data/data.csv' WITH HEADER = true;
SELECT * FROM users WHERE user_id = 8 AND activity = 'Standing' ORDER BY timestamp asc LIMIT 10;
```

###Data

The data as a cvs file are available in the directory ```/data/data.csv.zip```.
You can also find the features in the zip ```/data/features-actitracker.zip```.
