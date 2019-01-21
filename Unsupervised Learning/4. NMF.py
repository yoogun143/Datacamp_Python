import os
os.chdir('E:\Datacamp\Python\\Unsupervised Learning')
import pandas as pd
wiki = pd.read_csv('wiki.csv', header = None)
documents = wiki.loc[:,1].tolist()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
articles = tfidf.fit_transform(documents)
titles = wiki.loc[:,1]


#### NMF
# NMF fits to non-negative data only
# NMF is dimension reduction but models are interpretable (unlike PCA)

#### NMF TO TEXT
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)
df.head()

# Print row 7
df.iloc[7,:]
# => NMF features 2 has highest value => may represent topic

# Get the columns to tf-idf: words
words = tfidf.get_feature_names()

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)
# => there are 6 components just like PCA

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())
# => component 3 may relate to music topic
# for documents:
    # NMF components represent topics
    # NMF features combine topics into documents
    
# => from the initial tf-idf: articles with shape 463819 (documents) x 2107645 (terms), NMF reduce dimension to 463819 x 6 (features), the components with shape 6 x 2107645 is the same with the component - the axes in PCA
# => tf-idf = Multiply components by feature values, and add up


#### NMF FOR TEXT APPLICATION: RECOMMENDER SYSTEM I
# Recommend the articles that relate to the current articles
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the 8th document: article
article = df.iloc[7,:]

# Compute the dot products: similarities
# compare NMF feature values by cosine similarity: angles between the lines
# Different versions of the same document have same topic proporrions but exact feature values maybe different
# all versions lie on the same line
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())


#### NMF FOR TEXT APPLICATION: RECOMMENDER SYSTEM II
# Recommend music artists
# dataset:  rows correspond to artists and whose column correspond to users => find similaritiy point for 1 artist => if a user like A, he may like B
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# data
import scipy
data_artist = pd.read_csv('scrobbler-small-sample.csv')
data_artist2 = data_artist.pivot(index = 'artist_offset',
                                 columns = 'user_offset',
                                 values = 'playcount')
artists = data_artist2.fillna(0).values
artists = scipy.sparse.csr_matrix(artists)

# Create a MaxAbsScaler: scaler
# transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to.
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
artist_names = pd.read_csv('artists.csv').values
artist_names = [item for sublist in artist_names for item in sublist]
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Thanh Hoang']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())


#### NMF FOR IMAGES
# For images, NMF components are parts of images
# Read lcd data
samples = pd.read_csv('lcd-digits.csv')

# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples.loc[0,:]

# Print digit
print(digit)

# Use plt.imshow to display bitmap
def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

show_as_image(digit)

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)
# => digit normally has 104 pixel and now reduced to 7 components only
