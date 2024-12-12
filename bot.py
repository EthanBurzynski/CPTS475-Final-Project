import discord
from discord import app_commands
import json
import os
from dotenv import find_dotenv, load_dotenv
import datetime
from datetime import timezone, timedelta
import time
import asyncio
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from stumpy import stump
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

BOT_TOKEN = os.getenv("BOT_TOKEN")
MESSAGE_FILE_PATH = 'message_history_test.json'
ACTIVE_SERVER = 468638089359785984

# slash commands setup
discord.VoiceClient.warn_nacl = False
mintents = discord.Intents.all()
client = discord.Client(intents=mintents)
tree = app_commands.CommandTree(client)

# when bot is ready, prints the contents
@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(id=ACTIVE_SERVER))
    print("bot is ready")

CONCURRENT_CHANNEL_LIMIT = 10
semaphore = asyncio.Semaphore(CONCURRENT_CHANNEL_LIMIT)

@tree.command(
    name="collectdata",
    description="gets every message sent and sorts them, and then stores them",
    guild=discord.Object(id=ACTIVE_SERVER)
)
@app_commands.checks.has_permissions(administrator=True)
async def collectdata(interaction):
    # send a starter message
    await interaction.response.send_message(content=f'beginning message search', ephemeral=True)
    messageList = []
    server = client.get_guild(ACTIVE_SERVER)
    # create a task for every text channel in ACTIVE_SERVER
    taskList = [channelMessageCollector(chnl) for chnl in server.channels if str(chnl.type) == 'text']
    timeStart = time.time()
    # start running the tasks in taskList
    messageList = await asyncio.gather(*taskList)
    # flatten the list of lists into one single list
    allMessages = [msg for channelMsgs in messageList for msg in channelMsgs]
    timeEnd = time.time()
    print(f'{len(allMessages)} messages found and stored in allMessages, took {(timeEnd - timeStart):.2f} seconds')
    # begin sort
    timeStart = time.time()
    allMessagesSorted = sorted(allMessages, key = lambda msg: msg.created_at)
    timeEnd = time.time()
    print(f'{len(allMessagesSorted)} messages sorted, took {(timeEnd - timeStart):.2f} seconds')
    messageList.clear()
    # get all the essential information
    for msg in allMessagesSorted:
        msgDict = {
                "author": str(msg.author),
                "authorID": msg.author.id,
                "content": msg.content,
                "channel": str(msg.channel),
                "channelID": msg.channel.id,
                "msgID": msg.id,
                # subtracting 25200 because thats the number of seconds in 7 hours (utc to pst)
                "time": float(time.mktime((msg.created_at).timetuple()) - 25200)
        }
        messageList.append(msgDict)
    with open(MESSAGE_FILE_PATH, 'w') as f:
        for msg in messageList:
            json.dump(msg, f)
            f.write('\n')

async def channelMessageCollector(chnl):
    # make sure that no more processes than allowed are running
    async with semaphore:
        messageList = []
        timeList = []
        # only proceed with message gathering if the channel type is text
        timeStart = time.time()
        # run an initial history call to get the oldest 500 messages to establish the currentMessage variable
        async for msg in chnl.history(limit=500, oldest_first=True):
            messageList.append(msg)
        # set currentMessage to the most recently added message
        # could be a rare edge case where there's a channel with no messages
        currentMessage = messageList[-1] if messageList else None
        # once the last messages have been processed, currentMessage should be set to None
        while currentMessage:
            tempMessageList = []
            async for msg in chnl.history(limit=500, after=currentMessage, oldest_first=True):
                tempMessageList.append(msg)
            # exit loop if no messages were found
            if not tempMessageList:
                currentMessage = None
                timeEnd = time.time()
                timeList.append(timeEnd - timeStart)
            else:
                # add values from tempMessageList to messageList
                messageList.extend(tempMessageList)
                currentMessage = messageList[-1]
            # console info to amke sure everything is running
            if(len(messageList) % 3000 == 0):
                timeEnd = time.time()
                timeList.append(timeEnd - timeStart)
                print(f'channel: {chnl.name}, {len(messageList)}, took {timeList[-1]:.2f} seconds')
                timeStart = time.time()
        print(f'channel: {chnl.name} DONE, {len(messageList)} messages took {sum(timeList):.2f} seconds')
        return messageList

@tree.command(
    name="activityheatmap",
    description="generate a heatmap",
    guild=discord.Object(id=ACTIVE_SERVER)
)
@app_commands.describe(type = "heatmap or line")
async def activityheatmap(interaction, type: str = 'heatmap'):
    await interaction.response.send_message(f'.')
    # want to get the number of messages sent for every month
    # useful numbers: 86400 seconds is one day, 2628000 seconds is one month
    # start time, hardcoded to first message every sent
    startTime = datetime.datetime.fromtimestamp(1535451470)
    messagesInMonth = 0
    monthList = []
    with open(MESSAGE_FILE_PATH, 'r') as f:
        for line in f:
                # the everything variable
                z = json.loads(line)
                timeElapsed = z.get("time") - startTime.timestamp()
                # simple tracker to determine if a month in seconds has passed since the first marked message
                if(timeElapsed >= 2628000):  
                    # if a month has passed, append the number of counted messages to monthList and set the new startTime to the newest message
                    monthList.append(messagesInMonth)
                    messagesInMonth = 0
                    startTime = datetime.datetime.fromtimestamp(z.get("time"))
                else:
                    messagesInMonth += 1
    # optional field passed into the function to determine if data is displayed as a heatmap or a line graph
    if(type == 'heatmap'):
        # 12 columns, one for each month in a year. determine number of years based on number of months / 12
        cols = 12
        rows = math.ceil(len(monthList) / cols)
        # since i'm using math.ceil above to make sure its a rectangular heatmap, need to make sure the empty spots have 0 instead of nothing
        for x in range(len(monthList), rows * cols):
            monthList.append(0)
        # dirty line to reshape monthList into the heatmap shape
        # is this even necessary?
        hmData = np.array(monthList[:rows * cols]).reshape(rows, cols)
        # graph the data
        plt.title(f'Activity heatmap of messages sent by month')
        plt.imshow(hmData, cmap='cool', interpolation='nearest', aspect='equal')
        plt.colorbar()
        plt.xlabel("Months passed")
        plt.ylabel("Years passed")
        # add number of messages for each month into the heatmap cell
        i = 0
        for hmInt in monthList:
            plt.text(i % cols, i // cols, hmInt, ha='center', va='bottom', color='black', fontfamily='monospace', fontweight='bold', fontsize = 7)
            i += 1
    elif(type == 'line'):
        plt.figure(figsize = (12,6))
        plt.plot(range(len(monthList)), monthList, linewidth=2, label='Messages per Month')
        plt.title(f'Line graph of messages sent by month')
        plt.xlabel(f'Months passed')
        plt.ylabel(f'Number of messages sent')
        plt.grid(True)
    # send through discord bot
    filename = "activity.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    graph = discord.File(filename)
    embed = discord.Embed()
    embed.set_image(url="attachment://activity.png")
    await interaction.edit_original_response(embed=embed, attachments=[graph])

    return

@tree.command(
    name="timeaverage",
    description="unique words",
    guild=discord.Object(id=ACTIVE_SERVER)
)
async def timeaverage(interaction):
    await interaction.response.send_message(f'.')
    # get the data from the file
    messages = pd.read_json(MESSAGE_FILE_PATH, lines=True)
    # super messy line to convert the unix time to the hour of the day it was sent, accounting for time zones (i hate time zones.)
    messages['hour'] = messages['time'].apply(lambda x: datetime.datetime.fromtimestamp(x, tz=timezone(timedelta(hours=-8)))).dt.hour
    # create a list of the number of messages sent for each hour
    messagesByHour = messages['hour'].value_counts().sort_index()
    # create a graph using matplotlib
    plt.figure(figsize=(10, 6))
    plt.title(f'Messages sorted by time sent')
    plt.bar(messagesByHour.index, messagesByHour.values)
    # annoying little bit of code to fix the way the hours were formatted. also changes 0:00 to 24:00
    formattedHours = []
    for hour in messagesByHour.index:
        formattedHours.append(f'{hour if hour != 0 else 24}:00')
    # graph labels
    plt.xticks(messagesByHour.index, formattedHours, fontsize=6)
    plt.xlabel('Hour of the day')
    plt.ylabel('Total messages sent')
    plt.grid(axis='y', alpha=0.7)
    # send graph through discord bot
    filename = "timeaverage.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    graph = discord.File(filename)
    embed = discord.Embed()
    embed.set_image(url="attachment://timeaverage.png")
    await interaction.edit_original_response(embed=embed, attachments=[graph])

@tree.command(
    name="matrixprofile",
    description="matrix profile",
    guild=discord.Object(id=ACTIVE_SERVER)
)
async def viewmatrixprofile(interaction, type: int = 1):
    await interaction.response.send_message(f'.')
    # get the data
    messages = pd.read_json(MESSAGE_FILE_PATH, lines=True)
    # convert unix time back to datetime
    messages['datetime'] = pd.to_datetime(messages['time'], unit='s')
    # double check that it's sorted properly
    messages = messages.sort_values('datetime')

    # is there maybe another time series i can create with the data besides just message creation time?
    # messages per author per hour maybe? see what different people's habits are
    # how would i create a time series for the author thing though
    # cause there are a bunch of different authors
    # might be too annoying. ill skip it for now
    # could also do average message length per hour
    # how about all three

    # message frequency (number of messages sent per hour) time series
    messageFrequencyTimeSeries = messages.resample('d', on='datetime').size()
    # message length (average length of messages sent per hour) time series
    messages['messageLength'] = messages['content'].str.len()
    # there could be a day where no messages were sent and this will throw a divide by 0 error
    messageLengthTimeSeries = messages.resample('d', on='datetime')['messageLength'].mean()
    messageLengthTimeSeries = messageLengthTimeSeries.fillna(0)

    # stump needs a numpy array
    # segmenting which method to graph based on user input
    if(type == 1):
        messageFrequencyTimeSeriesNP = messageFrequencyTimeSeries.astype(np.float64)
        # looking for 180 day long subsequences
        freqM = 180
        yRange = 3000
        # create the matrix profiles
        # wow these really take a while to run huh
        frequencyMatrixProfile = stump(messageFrequencyTimeSeriesNP, freqM)
        motif_idx = np.argsort(frequencyMatrixProfile[:, 0])[0]
        nearest_neightbor_idx = frequencyMatrixProfile[motif_idx, 1]
    else:
        messageLengthTimeSeriesNP = messageLengthTimeSeries.astype(np.float64)
        # looking for 30 day long subsequences
        freqM = 180
        yRange = 450
        lengthMatrixProfile = stump(messageLengthTimeSeriesNP, freqM)
        motif_idx = np.argsort(lengthMatrixProfile[:, 0])[0]
        nearest_neightbor_idx = lengthMatrixProfile[motif_idx, 1]

    # using subplots to show both time series and matrix profile together
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), sharex=True, gridspec_kw={'hspace': 0})

    if(type == 1):
        ax1.plot(range(len(messageFrequencyTimeSeries)), messageFrequencyTimeSeries, linewidth=1)
        ax1.set_title('Message Frequency Time Series', fontsize='20')
        ax1.set_ylabel('Number of Messages', fontsize='15')
        rect = Rectangle((motif_idx, 0), freqM, yRange, facecolor='lightgrey')
        ax1.add_patch(rect)
        rect = Rectangle((nearest_neightbor_idx, 0), freqM, yRange, facecolor='lightgrey')
        ax1.add_patch(rect)

        ax2.plot(np.arange(len(frequencyMatrixProfile.P_)), frequencyMatrixProfile.P_, linewidth=1)
        ax2.set_title('Message Frequency Matrix Profile', fontsize='20')
        ax2.set_xlabel('Time, Segmented Daily', fontsize='15')
        ax2.set_ylabel('Distance between closest matching subsequence', fontsize='15')
        ax2.axvline(x=motif_idx, linestyle="dashed")
        ax2.axvline(x=nearest_neightbor_idx, linestyle="dashed")
    else:
        ax1.plot(range(len(messageLengthTimeSeries)), messageLengthTimeSeries, linewidth=1)
        ax1.set_title('Message Length Time Series', fontsize='20')
        ax1.set_ylabel('Average Message Length', fontsize='15')
        rect = Rectangle((motif_idx, 0), freqM, yRange, facecolor='lightgrey')
        ax1.add_patch(rect)
        rect = Rectangle((nearest_neightbor_idx, 0), freqM, yRange, facecolor='lightgrey')
        ax1.add_patch(rect)

        ax2.plot(np.arange(len(lengthMatrixProfile.P_)), lengthMatrixProfile.P_, linewidth=1)
        ax2.set_title('Message Length Matrix Profile', fontsize='20')
        ax2.set_xlabel('Time, Segmented Daily', fontsize='20')
        ax2.set_ylabel('Distance between closest matching subsequence', fontsize='15')
        ax2.axvline(x=motif_idx, linestyle="dashed")
        ax2.axvline(x=nearest_neightbor_idx, linestyle="dashed")
    plt.tight_layout()

    filename = "matrixprofile.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    graph = discord.File(filename)
    embed = discord.Embed()
    embed.set_image(url="attachment://matrixprofile.png")
    await interaction.edit_original_response(embed=embed, attachments=[graph])

@tree.command(
    name="sentiment",
    description="sentiment analysis for each person",
    guild=discord.Object(id=ACTIVE_SERVER)
)
async def sentiment(interaction):
    await interaction.response.send_message(f'.')
    # easy solution to block out bots and peopel w ho arent active
    # put it in the dotenv file for privacy
    allowedPeople = os.getenv("ALLOWED_PPL")
    # get the data
    messages = pd.read_json(MESSAGE_FILE_PATH, lines=True)
    # create the VADER analyzer
    analyzer = SentimentIntensityAnalyzer()
    # create new column for sentiment of the message
    # only grabbing the compound sentiment cause i dont want to bother with the other 3
    messages['sentiment'] = messages['content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    # messages.to_csv("messages.csv", index=False)
    
    # convert unix time back to datetime
    messages['datetime'] = pd.to_datetime(messages['time'], unit='s')
    # double check that it's sorted properly
    messages = messages.sort_values('datetime')
    # currently every message has a sentiment value
    # i want to group the data into weeks, where each author's sentiment values get averaged out for that week
    messages['week'] = messages['datetime'].dt.to_period('W')
    # print(messages['week'])
    # after looking through this data i think i'll need to add the number of messages sent by the author for that week to root out any outliers (for example, sending one really negative message and only that message in a one week period)
    sentimentData = messages.groupby(['author', 'week'])['sentiment'].mean().reset_index().sort_values(['author', 'week'])
    messageData = messages.groupby(['author', 'week'])['content'].count().reset_index()
    groupedData = pd.merge(sentimentData, messageData, on=['author', 'week'])
    # if number of messages sent in a week was less than 10, change sentiment to 0 (neutral)
    groupedData.loc[groupedData['content'] < 10, 'sentiment'] = 0
    # convert the pandas period into the start time of the period so i can plot it with matplotlib
    groupedData['week'] = groupedData['week'].dt.start_time
    groupedData.to_csv("groupedData.csv", index=False)
    # and then i can graph each author
    plt.figure(figsize=(16, 10))
    # get each unique person in the dataframe
    for person in groupedData['author'].unique():
        # get all the data from that person and load it into a new dataframe
        # print(person)
        personData = groupedData[groupedData['author'] == person]
        if person in allowedPeople:
            plt.plot(personData['week'], personData['sentiment'], alpha=0.8)
    allowedGroupedData = groupedData[groupedData['author'].isin(allowedPeople)]
    meanSentiment = allowedGroupedData.groupby('week')['sentiment'].mean()
    # this ends up reminding me a lot of a time series. if i had more time i would have explored the matrix profile of this
    plt.plot(meanSentiment.index, meanSentiment.values, label='Mean Sentiment', color='black', linewidth=2)

    plt.title('Sentiment Analysis For Each Person, Week by Week')
    plt.xlabel('Week')
    plt.ylabel('Average Sentiment')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.6, axis='y')
    plt.tight_layout()
    
    # modify x axis to show more information
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=90)

    filename = "sentiment.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    graph = discord.File(filename)
    embed = discord.Embed()
    embed.set_image(url="attachment://sentiment.png")
    await interaction.edit_original_response(embed=embed, attachments=[graph])

client.run(BOT_TOKEN)
