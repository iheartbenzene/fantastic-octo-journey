#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:00:38 2019

@author: jjormungand
"""

import pycard
import pandas as pd

# Create a card

df_source = pd.read_excel('', sheet_name = 0)

credit_card = df_source.copy()

credit_card.columns

credit_card.head()

credit_card.columns.to_series().groupby(credit_card.dtypes).groups

credit_card.info() 

#Add assetion to test that info() has no missing values.

card = pycard.Card(
    number='4444333322221111',
    month=1,
    year=2020,
    cvc=123
)

# Validate the card (checks that the card isn't expired and is mod10 valid)
assert card.is_valid

# Perform validation checks individually
assert not card.is_expired
assert card.is_mod10_valid

# The card is a visa
assert card.brand == 'visa'
assert card.friendly_brand == 'Visa'
assert card.mask == 'XXXX-XXXX-XXXX-1111'

# The card is a known test card
assert card.is_test