import random #for selecting random answer
import json
import torch
from model import NeuralNet
from pre_processing import bag_of_words, tokenize
import datetime
import pymongo
import uuid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

with open('corpus.json', 'r') as f:
    corpus = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "KFCbot"

seat_count = 50
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Restaurant"]
menu_collection = db["menu"]
feedback_collection = db["feedback"]
bookings_collection = db["bookings"]
with open("corpus.json") as file:
    data_f = json.load(file)


#Seats reduce by 1 and customer gets unique booking_id and is updated on Database

def book_table():
    global seat_count
    seat_count = seat_count - 1
    booking_id = str(uuid.uuid4())
    now = datetime.datetime.now()
    booking_time = now.strftime("%Y-%m-%d %H:%M:%S")
    booking_doc = {"booking_id": booking_id, "booking_time": booking_time}
    bookings_collection.insert_one(booking_doc)
    return booking_id


def vegan_menu():
    query = {"vegan": "Y"}
    vegan_doc = menu_collection.find(query)
    if vegan_doc.count() > 0:
        response = "Vegan options are: "
        for x in vegan_doc:
            response = response + str(x.get("item")) + " for Ksh. " + str(x.get("cost")) + "; "
        response = response[:-2] # to remove the last ;
    else:
        response = "Sorry no vegan options are available"
    return response


def veg_menu():
    query = {"veg": "Y"}
    vegan_doc = menu_collection.find(query)
    if vegan_doc.count() > 0:
        response = "Vegetarian options are: "
        for x in vegan_doc:
            response = response + str(x.get("item")) + " for Ksh. " + str(x.get("cost")) + "; "
        response = response[:-2] # to remove the last ;
    else:
        response = "Sorry no vegetarian options are available"
    return response


def offers():
    all_offers = menu_collection.distinct('offer')
    if len(all_offers)>0:
        response = "The SPECIAL OFFERS are: "
        print("\n")
        for ofr in all_offers:
            docs = menu_collection.find({"offer": ofr})
            response = response + ' ' + ofr.upper() + " On: "
            for x in docs:
                response = response + str(x.get("item")) + " - Ksh. " + str(x.get("cost")) + "; "
                print("\n")
            response = response[:-2] # to remove the last ;
    else:
        response = "Sorry there are no offers available now."
    return response


def suggest():
    day = datetime.datetime.now()
    day = day.strftime("%A")
    if day == "Monday":
        response = "Chef recommends: Paneer Grilled Roll, Jade Chicken"
    elif day == "Tuesday":
        response = "Chef recommends: Tofu Cutlet, Chicken A La King"

    elif day == "Wednesday":
        response = "Chef recommends: Mexican Stuffed Bhetki Fish, Crispy corn"

    elif day == "Thursday":
        response = "Chef recommends: Mushroom Pepper Skewers, Chicken cheese balls"

    elif day == "Friday":
        response = "Chef recommends: Veggie Steak, White Sauce Veggie Extravaganza"

    elif day == "Saturday":
        response = "Chef recommends: Tofu Cutlet, Veggie Steak"

    elif day == "Sunday":
        response = "Chef recommends: Chicken Cheese Balls, Butter Garlic Jumbo Prawn"
    return response


def recipe_enquiry(msg):
    all_foods = menu_collection.distinct('item')
    response = ""
    for food in all_foods:
        query = {"item": food}
        food_doc = menu_collection.find(query)[0]
        if food.lower() in msg.lower():
            response = food_doc.get("about")
            break
    if "" == response:
        response = "Sorry please try again with exact spelling of the food item!"
    return response


def record_feedback(msg, type):
    feedback_doc = {"feedback_string": msg, "type": type}
    feedback_collection.insert_one(feedback_doc)


def get_specific_response(tag):
    for intent in data_f['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
    response = random.choice(responses)
    return response


def show_menu():
    all_items = menu_collection.distinct('item')
    response = ', '.join(all_items)
    return response


def get_response(msg):
    global seat_count
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in corpus["intents"]:
            if tag == intent["tag"]:
                response = ""
                if tag != "":
                    if tag == "book_table":

                        if seat_count > 0:
                            booking_id = book_table()
                            response = "Your table has been booked successfully. Please show this Booking ID at the counter: " + str(
                                booking_id)
                        else:
                            response = "Sorry we are sold out now!"


                    elif tag == "available_tables":
                        response = "There are " + str(seat_count) + " table(s) available at the moment."

                    elif tag == "veg_enquiry":
                        response = veg_menu()

                    elif tag == "vegan_enquiry":
                        response = vegan_menu()

                    elif tag == "offers":
                        response = offers()

                    elif tag == "suggest":
                        response = suggest()

                    elif tag == "recipe_enquiry":
                        response = recipe_enquiry(msg)

                    elif tag == "menu":
                        response = show_menu()

                    elif tag == "positive_feedback":
                        record_feedback(msg, "positive")
                        response = "Thank you so much for your valuable feedback. We look forward to serving you again!"

                    elif  tag == "negative_feedback":
                        record_feedback(msg, "negative")
                        response = "Thank you so much for your valuable feedback. We deeply regret the inconvenience. We have " \
                                "forwarded your concerns to the authority and hope to satisfy you better the next time! "
                    # for other intents with pre-defined responses that can be pulled from dataset
                    else:
                        response = get_specific_response(tag)
                else:
                    response = "Sorry! I didn't get it, please try to be more precise."
                return response


