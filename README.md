
# Machine learning technical test

Hello there! 👋🏼 

As mentioned in the interview, the first area of our system that we will expect you to be working on is around receipts. Therefore, we’ve come up with some scenarios that will be relevant to your initial tasks. 

Please choose one from the below scenarios to build. 

---

**Scenario 1: Retailer name prediction**

Reliably predict retailer based on receipt ocr raw textual data. 

Please use the provided dataset, in particular the `establishment` in the `rawData` field associated to each receipt. 

---

**Scenario 2: Retailer name detection**

Categorise retailer based on receipt image. 

We want you predict the receipt retailer name and categorise into the following categories:

* Boots

* Holland & Barrett

* No category

Please filter the provided dataset to get the data associated with above retailer names (hint: Boots and Boots UK Limited are actually the same retailer). To get the receipt image in order to complete the task, please use the `receiptImage` field associated to each receipt.

---

**Scenario 3: Receipt upload volume prediction**

Predict volume of receipts that will be uploaded per day for the next 4 weeks.

Please use the provided dataset and use the `uploadedTime` field associated to each receipt.

---

**Bonus section**

This section is optional in case you complete before the alotted time period.

* Your solution works on a mobile device (i.e. we can run and get instant results on mobile)

* Any extensions to the use-cases that you feel would be interesting

---

## Which dataset should I use?

Please see the provided .csv file in this repo. Some notes to give you a bit of context on the data:

**purchaseTime:** when the receipt items purchase was made

**uploadedTime:** when the receipt image was uploaded from the app

**lastUpdated:** the last time any data was changed

**status:**

- REVIEWED = admin user has verified it is a valid receipt

- DELETED = admin or user has removed the receipt 

- REJECTED = admin user has defined it is an invalid receipt

- NEEDS_REVIEW = ocr has completed and is awaiting admin user approval

**retailerName:** the name we’ve mapped to the retailer after applying a basic algorithm to it

**total:** the total amount value of the receipt after admin user has verified 

**totalTax:** the total tax value of the receipt after admin user has verified 

**rawData:** the raw text response from our ocr provider

**totalConfidence:** the confidence that our ocr provider thinks the rawData is correct

**receiptImage:** url to the image the user has uploaded of the receipt

## What are the deliverables?

1. Your code for one of the scenarios.

2. A small write up explaining: the reasoning behind your choice, discussion of the results and performance of your solution and also areas where it can be improved. Please keep to a maximum of 500 words.

## Where should I submit the test?

Please use this git repository to submit the test.

## How long should I spend on this?

To make the assessment fair for all candidates, please spend no longer than 4 hours on this task. 

If you haven’t finished, please leave some documentation explaining what you would have done next.

## When does it need to be completed by?

The latest date you should submit the test by is 14th July, as we will commence technical interviews the week after.

## How will I be assessed?

1. Functionality  
    a. Can we run the system?  
    b. Does it provide a meaningful output?  
    c. Is there any further functionality or creative ideas shown?

2. Code quality   
    a. Readable?  
    b. Maintainable?  
    c. Use of relevant frameworks?  

3. Write up  
    a. Clear instructions on how to test system.  
    b. Concise explanation of system and why you chose the idea.  
    c. Presentation of results.  
    d. Notes on any improvements you’d make to the system.  

