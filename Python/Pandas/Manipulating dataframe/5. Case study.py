import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Manipulating dataframe')
medals = pd.read_csv('all_medalists.csv')


#### TOP COUNTRIES BY NUMBER OF MEDALS
# Select the 'NOC' column of medals: country_names
country_names = medals["NOC"]

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Name of countries
country_names.unique()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))


#### COUNT MEDALS BY TYPE
# Construct the pivot table: counted
counted = medals.pivot_table(index="NOC", columns="Medal", values="Athlete", aggfunc="count")

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis="columns")

# Sort counted by the 'totals' column
counted = counted.sort_values("totals", ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))


#### SHOULD WE DROP ONE OF THE COLUMN EVENT_GENDER AND GENDER?
# Select columns: ev_gen
ev_gen = medals[["Event_gender", "Gender"]]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)


#### FIND ERROR WITH ABOVE TABLE
# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(["Event_gender", "Gender"])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)
# => only one suspicious row: This is likely a data error.


#### WHERE IS THAT SUSPICIOUS DATA?
# Create the Boolean Series: sus
sus = (medals.Event_gender == "W") & (medals.Gender == "Men")

# Create a DataFrame with the suspicious row: suspect
suspect = medals[sus]

# Print suspect
print(suspect)


#### WHICH COUNTRIES WON MEDALS IN THE MOST DISTINCT SPORTS?
# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby("NOC")

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped["Sport"].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))


#### NUMBER OF DISTINCT SPORTS IN WHICH USA AND USSR WON MEDALS DURING COLD WAR
# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war
during_cold_war = (medals["Edition"] >= 1952) & (medals["Edition"] <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(["USA", "URS"])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby("NOC")

# Create Nsports
Nsports = country_grouped["Sport"].nunique().sort_values(ascending=False)

# Print Nsports
print(Nsports)


#### USA OR USSR WON THE MOST MEDALS CONSISTENTLY OVER COLD WAR?
# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index="Edition", columns="NOC", values="Athlete", aggfunc="count")

# Slice medals_won_by_country: cold_war_usa_usr_medals
cold_war_usa_usr_medals = medals_won_by_country.loc[1952:1988, ["USA","URS"]]

# Create most_medals 
most_medals = cold_war_usa_usr_medals.idxmax(axis="columns")

# Print most_medals.value_counts()
print(most_medals.value_counts())


#### USA MEDAL COUNTS BY EDITION
# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values=medals.Medal, categories=["Bronze", "Silver", "Gold"], ordered=True)

# Create the DataFrame: usa
usa = medals[(medals["NOC"] == "USA")]

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(["Edition", "Medal"]).Athlete.count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level="Medal")

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
