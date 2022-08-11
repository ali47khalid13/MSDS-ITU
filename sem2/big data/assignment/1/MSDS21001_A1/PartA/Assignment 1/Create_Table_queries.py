# DROP TABLES
# Write queries to drop each table, please don't change variable names,
# You should write respective queries against each varibale

Videoplay_table_drop = "DROP TABLE IF EXISTS videoplay_fact"
Users_table_drop = "DROP TABLE IF EXISTS users_dim"
Videos_table_drop = "DROP TABLE IF EXISTS videos_dim"
Youtubers_table_drop = "DROP TABLE IF EXISTS youtubers_dim"
Time_table_drop = "DROP TABLE IF EXISTS time_dim"

# CREATE TABLES
# Write queries to create each table, please don't change variable names, you can refer to star schema table
# You should write respective queries against each varibale

Users_table_create = ("""CREATE TABLE IF NOT EXISTS Users_dim (user_id text, 
                                                               first_name text, 
                                                               last_name text, 
                                                               gender text, 
                                                               level text, 
                        PRIMARY KEY (user_id));""")

Youtubers_table_create = ("""CREATE TABLE IF NOT EXISTS youtubers_dim ( youtuber_id text,
                                                                        name text, 
                                                                        location text, 
                                                                        latitude text, 
                                                                        longitude text, 
                        PRIMARY KEY (youtuber_id));""")

Videos_table_create = ("""CREATE TABLE IF NOT EXISTS videos_dim (video_id text, 
                                                                 title text, 
                                                                 youtuber_id text, 
                                                                 year text, 
                                                                 duration text, 
                        PRIMARY KEY (video_id),
                        CONSTRAINT fk_youtuber FOREIGN KEY(youtuber_id) REFERENCES youtubers_dim(youtuber_id));""")



Time_table_create = ("""CREATE TABLE IF NOT EXISTS time_dim (start_time timestamp, 
                                                             hour text, 
                                                             day text, 
                                                             week text, 
                                                             month text, 
                                                             year text, 
                                                             weekday text, 
                        PRIMARY KEY (start_time));""")

Videoplay_table_create = ("""CREATE TABLE IF NOT EXISTS videoplay_fact (videoplay_id serial, 
                                                                        start_time timestamp, 
                                                                        user_id text, 
                                                                        level text, 
                                                                        video_id text, 
                                                                        youtuber_id text, 
                                                                        session_id text, 
                                                                        location text, 
                                                                        user_agent text, 
                            PRIMARY KEY(videoplay_id),
                            CONSTRAINT fk_time FOREIGN KEY(start_time) REFERENCES time_dim(start_time),
                            CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users_dim(user_id), 
                            CONSTRAINT fk_video FOREIGN KEY(video_id) REFERENCES videos_dim(video_id),
                            CONSTRAINT fk_youtuber FOREIGN KEY(youtuber_id) REFERENCES youtubers_dim(youtuber_id));""")



# INSERT RECORDS
# Write queries to insert record to each table, please don't change variable names, you can refer to star schema table
# You should write respective queries against each varibale

Videoplay_table_insert = ("""INSERT INTO videoplay_fact (start_time,user_id,level,video_id,youtuber_id,session_id,location,user_agent)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""")

Videos_table_insert = ("""INSERT INTO videos_dim (Video_id, title, youtuber_id, year, duration)
                        VALUES (%s, %s, %s, %s, %s)""")

Users_table_insert = ("""INSERT INTO users_dim (user_id, first_name, last_name, gender, level)
                        VALUES (%s, %s, %s, %s, %s)""")

Time_table_insert = ("""INSERT INTO time_dim (start_time, hour, day, week, month, year, weekday)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)""")

Youtubers_table_insert = ("""INSERT INTO youtubers_dim (youtuber_id, name, location, latitude, longitude)
                        VALUES (%s, %s, %s, %s, %s)""")

# QUERY LISTS

create_table_queries = [Users_table_create, Time_table_create, Youtubers_table_create, Videos_table_create, Videoplay_table_create]
drop_table_queries = [Videoplay_table_drop, Users_table_drop, Videos_table_drop, Youtubers_table_drop, Time_table_drop]
