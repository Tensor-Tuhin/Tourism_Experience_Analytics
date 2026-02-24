import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_tourism_trends_year(df):
    # Calculating the number of unique tourists for every year
    tpy = df.groupby('VisitYear')['UserId'].count().reset_index()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(data=tpy, x='VisitYear', y='UserId', palette='inferno', ax=ax)
    ax.set_title('Number of Unique Tourists per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Tourist Count')
    return fig


def plot_tourism_trends_month(df):
    # To keep the months in the correct order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
    
    # Calculting the number of tourists for every month
    tpm = df.groupby('Month')['UserId'].count().sort_values(ascending=False).reset_index()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(data=tpm, x='Month', y='UserId', palette='Set1', ax=ax)
    ax.set_title('Number of Unique Tourists per Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Tourist Count')
    plt.xticks(rotation=45)
    return fig


def plot_rating_distribution(df):
    # Plotting the histogram
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df['Rating'], bins=30, kde=True, color='lightseagreen', ax=ax)
    ax.set_title('Distribution of Ratings')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Frequency')
    return fig


def plot_avg_rating_by_attr_type(df):
    # Calculating the average rating by attraction type
    avg_attr_rating = df.groupby('AttractionType')['Rating'].mean().reset_index().sort_values(ascending=False, by='Rating')

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=avg_attr_rating, x='AttractionType', y='Rating', palette='viridis', ax=ax)
    ax.set_title('Average Rating by Attraction Type')
    ax.set_xlabel('Attraction Type')
    ax.set_ylabel('Average Rating')
    plt.xticks(rotation=90)
    return fig


def plot_count_visitmode(df):
    # Calculating the count of each visit mode
    visit_mode_count = df['VisitMode'].value_counts().reset_index()
    visit_mode_count.columns = ['VisitMode', 'Count']

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(data=visit_mode_count, x='VisitMode', y='Count', palette='tab10', ax=ax)
    ax.set_title('Count of Each Visit Mode')
    ax.set_xlabel('Visit Mode')
    ax.set_ylabel('Count')
    return fig


def plot_avg_type_tourist_rating(df):
    # Calculating the average rating given by each tourist type
    avg_ratings = df.groupby('VisitMode')['Rating'].mean().reset_index()
    
    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(data=avg_ratings, x='VisitMode', y='Rating', color='g', marker='o', ax=ax)
    ax.set_title('Average Rating by each Tourist type')
    ax.grid(color='y')
    ax.set_xlabel('Tourist Type')
    ax.set_ylabel('Average Rating')
    return fig


def plot_prfrd_attr_by_tourist_type(df):
    # Calculating the most visited attraction type for each tourist type
    preferred_attr = df.groupby(['VisitMode', 'AttractionType']).size().reset_index(name='Count')
    preferred_attr = preferred_attr.sort_values(ascending=False, by='Count').groupby('VisitMode').first().reset_index()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(data=preferred_attr, x='VisitMode', y='Count', hue='AttractionType', palette='Dark2', ax=ax)
    ax.set_title('Most Visited Attraction Type by Tourist Type')
    ax.set_xlabel('Tourist Type')
    ax.set_ylabel('Number of Attraction Types Visited')
    ax.legend(title='Attraction Type')
    return fig


def plot_mode_by_month(df):
    # To keep the months in the correct order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
    
    # Calculating the count of each visit mode for each month
    count = pd.crosstab(df['VisitMode'], df['Month'])

    # Converting the count to percentage
    pct = count.div(count.sum(axis=1), axis=0)*100

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(pct, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax)
    ax.set_title('Visit Mode Distribution by Month (percentage)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Visit Mode')
    return fig


def plot_top_ten_attr(df):
    # Calculating the top 10 most visited attraction types
    top_attr = df['Attraction'].value_counts().head(10).reset_index()
    top_attr.columns = ['Attraction', 'Count']

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(data=top_attr, x='Attraction', y='Count', palette='Set2', ax=ax)
    ax.set_title('Top 10 Most Visited Attractions')
    ax.set_xlabel('Attraction')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90)
    return fig


def plot_user_interactions(df):
    # Calculating the number of interactions for each user
    user_interactions = df.groupby('UserId').size()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(user_interactions, bins=30, color='coral', log_scale=True, kde=True, ax=ax)
    ax.set_title('Distribution of User Interactions')
    ax.set_xlabel('Number of Interactions')
    ax.set_ylabel('Frequency (No. of Users)')
    return fig


def plot_attr_interactions(df):
    # Calculating the number of interactions for each attraction
    attr_interactions = df.groupby('Attraction').size()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(attr_interactions, bins=30, color='steelblue', kde=True, ax=ax)
    ax.set_title('Distribution of Attraction Interactions')
    ax.set_xlabel('Number of Interactions')
    ax.set_ylabel('Frequency (No. of Attractions)')
    return fig

def recommend_similar_attractions(attraction, attr_similarity_df, top_n=5):
    if attraction not in attr_similarity_df.columns:
        return f"{attraction} not found"
    
    # Get the similarity scores for the given attraction
    similarity_scores = attr_similarity_df[attraction]

    # Sort the scores in descending order
    similar_attr = similarity_scores.sort_values(ascending=False)[1:top_n+1]

    # Return top_n similar attractions (excluding the attraction itself)
    return similar_attr.index.tolist()