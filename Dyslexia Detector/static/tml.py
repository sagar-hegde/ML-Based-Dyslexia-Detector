import json
import os
import uuid
from datetime import datetime
import qrcode

DATA_FILE = "tickets.json"

MUSEUMS = {
    1: "National History Museum",
    2: "Art Gallery Museum",
    3: "Science Museum",
    4: "Heritage Museum"
}

def load_tickets():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_tickets(tickets):
    with open(DATA_FILE, "w") as f:
        json.dump(tickets, f, indent=4)

def login():
    print("\n===== MeseoTicket Login =====")
    username = input("Enter Username: ")
    password = input("Enter Password: ")
    print("Login Successful!\n")
    return username

def select_museum():
    print("Available Museums:")
    for key, value in MUSEUMS.items():
        print(f"{key}. {value}")
    choice = int(input("Select Museum: "))
    return MUSEUMS.get(choice, "Unknown Museum")

def generate_qr(ticket_id):
    img = qrcode.make(ticket_id)
    filename = f"{ticket_id}.png"
    img.save(filename)
    return filename

def book_ticket(user):
    tickets = load_tickets()

    museum = select_museum()
    visitors = int(input("Number of Visitors: "))
    visit_date = input("Visit Date (YYYY-MM-DD): ")

    ticket_id = str(uuid.uuid4())[:8]

    qr_file = generate_qr(ticket_id)

    ticket = {
        "Ticket ID": ticket_id,
        "User": user,
        "Museum": museum,
        "Visitors": visitors,
        "Visit Date": visit_date,
        "Booked On": str(datetime.now()),
        "QR Code": qr_file
    }

    tickets.append(ticket)
    save_tickets(tickets)

    print("\n===== Ticket Booked Successfully =====")
    print("Ticket ID :", ticket_id)
    print("Museum    :", museum)
    print("Visitors  :", visitors)
    print("Visit Date:", visit_date)
    print("QR Saved  :", qr_file)

def view_tickets(user):
    tickets = load_tickets()

    print("\n===== My Tickets =====")

    found = False

    for ticket in tickets:
        if ticket["User"] == user:
            found = True
            print("-" * 40)
            print("Ticket ID :", ticket["Ticket ID"])
            print("Museum    :", ticket["Museum"])
            print("Visitors  :", ticket["Visitors"])
            print("Visit Date:", ticket["Visit Date"])
            print("QR Code   :", ticket["QR Code"])

    if not found:
        print("No Tickets Found.")

def main():
    user = login()

    while True:
        print("\n====== MeseoTicket ======")
        print("1. Book Ticket")
        print("2. View My Tickets")
        print("3. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":
            book_ticket(user)

        elif choice == "2":
            view_tickets(user)

        elif choice == "3":
            print("\nThank you for using MeseoTicket!")
            print("Paperless • Contactless • Eco-Friendly")
            break

        else:
            print("Invalid Choice!")


if __name__ == "__main__":
    main()



def loadData(dataname):
    filename = home_dir + 'Resource/' + dataname + '.npz'
    npzfile = numpy.load(filename)
    print(npzfile.files)
    X = npzfile['X']
    y = npzfile['y']
    n, d = X.shape
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    return X, y

def experiment(xMat, yVec, maxiter, repeat, gamma, isSearch, isExact, newtoniter=100):
    demo = Demo(maxiter, repeat, gamma)
    demo.fit(xMat, yVec, m=256)
    condnum = demo.condnum
    print('Condition number is ' + str(condnum))

    m = 4
    print('m = ' + str(m))
    err1 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 16
    print('m = ' + str(m))
    err2 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 64
    print('m = ' + str(m))
    err3 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 256
    print('m = ' + str(m))
    err4 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)

    return err1, err2, err3, err4, condnum


def plotConvergence(outfilename, imagename):
    npzfile = numpy.load(outfilename)
    dataname = str(npzfile['dataname'])
    maxiter = npzfile['maxiter']
    newtoniter = npzfile['newtoniter']
    err1 = npzfile['err1']
    err2 = npzfile['err2']
    err3 = npzfile['err3']
    err4 = npzfile['err4']

    repeat = err1.shape[0]

    fig = plt.figure(figsize=(9, 8))

    for r in range(repeat):
        plt.semilogy(err1[r, :], color='greenyellow', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err2[r, :], color='salmon', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err3[r, :], color='skyblue', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err4[r, :], color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    line0, = plt.semilogy(numpy.median(err1, axis=0), color='g', linestyle='-', linewidth=3)
    line1, = plt.semilogy(numpy.median(err2, axis=0), color='r', linestyle='-', linewidth=3.5)
    line2, = plt.semilogy(numpy.median(err3, axis=0), color='b', linestyle='-', linewidth=3.5)
    line3, = plt.semilogy(numpy.median(err4, axis=0), color='k', linestyle='-', linewidth=3.5)

    fontsize = 32
    plt.xlabel('iterations (t)', fontsize=fontsize+2)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.axis([0, maxiter, 1e-8, 5])
    plt.tight_layout()

    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)



class ETicketSystem:
    def __init__(self):
        self.tickets = {}
        self.ticket_no = 1001

    def book_ticket(self):
        print("\n--- Book Ticket ---")
        name = input("Enter Name: ")
        destination = input("Enter Destination: ")
        seats = int(input("Enter Number of Seats: "))

        ticket = {
            "Name": name,
            "Destination": destination,
            "Seats": seats
        }

        self.tickets[self.ticket_no] = ticket
        print(f"\nTicket Booked Successfully!")
        print(f"Ticket Number: {self.ticket_no}")
        self.ticket_no += 1

    def view_tickets(self):
        print("\n--- Booked Tickets ---")

        if not self.tickets:
            print("No tickets found.")
            return

        for number, details in self.tickets.items():
            print("-" * 30)
            print("Ticket No :", number)
            print("Name      :", details["Name"])
            print("Destination:", details["Destination"])
            print("Seats     :", details["Seats"])

    def cancel_ticket(self):
        print("\n--- Cancel Ticket ---")
        number = int(input("Enter Ticket Number: "))

        if number in self.tickets:
            del self.tickets[number]
            print("Ticket Cancelled Successfully.")
        else:
            print("Ticket Not Found.")

    def search_ticket(self):
        print("\n--- Search Ticket ---")
        number = int(input("Enter Ticket Number: "))

        if number in self.tickets:
            t = self.tickets[number]
            print("\nTicket Found")
            print("Name:", t["Name"])
            print("Destination:", t["Destination"])
            print("Seats:", t["Seats"])
        else:
            print("Ticket Not Found.")

    def menu(self):
        while True:
            print("\n===== E-Ticketing System =====")
            print("1. Book Ticket")
            print("2. View Tickets")
            print("3. Search Ticket")
            print("4. Cancel Ticket")
            print("5. Exit")

            choice = input("Enter Choice: ")

            if choice == "1":
                self.book_ticket()

            elif choice == "2":
                self.view_tickets()

            elif choice == "3":
                self.search_ticket()

            elif choice == "4":
                self.cancel_ticket()

            elif choice == "5":
                print("Thank you for using the E-Ticketing System!")
                break

            else:
                print("Invalid Choice! Try Again.")


def main():
    system = ETicketSystem()
    system.menu()


if __name__ == "__main__":
    main()



import json
import os
import random
from datetime import datetime

USERS_FILE = "users.json"
TICKETS_FILE = "tickets.json"

def initialize_files():
    if not os.path.exists(USERS_FILE):
        admin = {
            "admin": {
                "password": "admin123",
                "role": "admin"
            }
        }
        with open(USERS_FILE, "w") as f:
            json.dump(admin, f, indent=4)

    if not os.path.exists(TICKETS_FILE):
        with open(TICKETS_FILE, "w") as f:
            json.dump([], f)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def load_tickets():
    with open(TICKETS_FILE, "r") as f:
        return json.load(f)

def save_tickets(tickets):
    with open(TICKETS_FILE, "w") as f:
        json.dump(tickets, f, indent=4)

def register():
    users = load_users()

    print("\n===== REGISTER =====")

    username = input("Username: ")

    if username in users:
        print("Username already exists.")
        return

    password = input("Password: ")

    users[username] = {
        "password": password,
        "role": "user"
    }

    save_users(users)

    print("Registration Successful!")

def login():

    users = load_users()

    print("\n===== LOGIN =====")

    username = input("Username: ")
    password = input("Password: ")

    if username in users:

        if users[username]["password"] == password:

            print("Login Successful")

            return username, users[username]["role"]

    print("Invalid Login")

    return None, None

def generate_ticket_id():
    return "T" + str(random.randint(10000,99999))

def user_menu(username):

    while True:

        print("\n===== USER MENU =====")
        print("1. Book Ticket")
        print("2. View My Tickets")
        print("3. Search Ticket")
        print("4. Cancel Ticket")
        print("5. Logout")

        choice = input("Enter Choice : ")

        if choice == "1":
            print("Booking Module Coming in Part 2")

        elif choice == "2":
            print("View Module Coming in Part 2")

        elif choice == "3":
            print("Search Module Coming in Part 3")

        elif choice == "4":
            print("Cancel Module Coming in Part 3")

        elif choice == "5":
            break

        else:
            print("Invalid Choice")

def admin_menu():

    while True:

        print("\n===== ADMIN MENU =====")
        print("1. View All Tickets")
        print("2. Reports")
        print("3. Logout")

        choice = input("Choice : ")

        if choice == "1":
            print("Admin Ticket View Coming in Part 5")

        elif choice == "2":
            print("Reports Coming in Part 5")

        elif choice == "3":
            break

        else:
            print("Invalid Choice")

def main():

    initialize_files()

    while True:

        print("\n==============================")
        print("   E - TICKETING SYSTEM")
        print("==============================")
        print("1. Register")
        print("2. Login")
        print("3. Exit")

        choice = input("Enter Choice : ")

        if choice == "1":
            register()

        elif choice == "2":

            username, role = login()

            if username:

                if role == "admin":
                    admin_menu()

                else:
                    user_menu(username)

        elif choice == "3":

            print("Thank You")
            break

        else:

            print("Invalid Choice")

if __name__ == "__main__":
    main()



def main(NewtonIter, Gamma, ResultName):
    dataname = 'covtype'
    path = home_dir + 'Output/logis/'
    MaxIter = 30
    Repeat = 10
    IsSearch = False
    IsExact = False

    xMat, yVec = loadData(dataname)
    print(xMat)
    print(yVec)

    err1, err2, err3, err4, condnum = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact, newtoniter=NewtonIter)
    outfilename = path + dataname + ResultName + '.npz'
    numpy.savez(outfilename, err1=err1, err2=err2, err3=err3, err4=err4, dataname=dataname, maxiter=MaxIter, newtoniter=NewtonIter, condnum=condnum)

    imagename = path + dataname + ResultName + '.pdf'
    plotConvergence(outfilename, imagename)

def loadData(dataname):
    filename = home_dir + 'Resource/' + dataname + '.npz'
    npzfile = numpy.load(filename)
    print(npzfile.files)
    X = npzfile['X']
    y = npzfile['y']
    n, d = X.shape
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    return X, y

def experiment(xMat, yVec, maxiter, repeat, gamma, isSearch, isExact, newtoniter=100):
    demo = Demo(maxiter, repeat, gamma)
    demo.fit(xMat, yVec, m=256)
    condnum = demo.condnum
    print('Condition number is ' + str(condnum))

    m = 4
    print('m = ' + str(m))
    err1 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 16
    print('m = ' + str(m))
    err2 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 64
    print('m = ' + str(m))
    err3 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 256
    print('m = ' + str(m))
    err4 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)

    return err1, err2, err3, err4, condnum


def plotConvergence(outfilename, imagename):
    npzfile = numpy.load(outfilename)
    dataname = str(npzfile['dataname'])
    maxiter = npzfile['maxiter']
    err1 = npzfile['err1']
    err2 = npzfile['err2']
    err3 = npzfile['err3']
    err4 = npzfile['err4']

    repeat = err1.shape[0]

    condnum = npzfile['condnum']
    print('Condition number is ' + str(condnum))

    fig = plt.figure(figsize=(9, 8))

    for r in range(repeat):
        plt.semilogy(err1[r, :], color='greenyellow', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err2[r, :], color='salmon', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err3[r, :], color='skyblue', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err4[r, :], color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    line0, = plt.semilogy(numpy.median(err1, axis=0), color='g', linestyle='-', linewidth=3)
    line1, = plt.semilogy(numpy.median(err2, axis=0), color='r', linestyle='-', linewidth=3.5)
    line2, = plt.semilogy(numpy.median(err3, axis=0), color='b', linestyle='-', linewidth=3.5)
    line3, = plt.semilogy(numpy.median(err4, axis=0), color='k', linestyle='-', linewidth=3.5)

    fontsize = 32
    plt.xlabel('iterations (t)', fontsize=fontsize+2)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.axis([0, 20, 1e-12, 5])
    plt.tight_layout()

    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)

def main(NewtonIter, Gamma, ResultName):
    dataname = 'logis_N8'
    path = home_dir + 'Output/logis/'
    MaxIter = max(20, int(numpy.ceil(1000 / (NewtonIter+10))))
    Repeat = 10
    IsSearch = True
    IsExact = False

    xMat, yVec = loadData(dataname)
    print(xMat)
    print(yVec)

    err1, err2, err3, err4, condnum = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact, newtoniter=NewtonIter)
    outfilename = path + dataname + ResultName + '.npz'
    numpy.savez(outfilename, err1=err1, err2=err2, err3=err3, err4=err4, dataname=dataname, maxiter=MaxIter, newtoniter=NewtonIter, condnum=condnum)

    imagename = path + dataname + ResultName + '.pdf'
    plotConvergence(outfilename, imagename)


if __name__ == '__main__':
    Gamma = 1e-6
    GammaName = '_gamma-6'

    NewtonIter = 270
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

    Gamma = 1e-8
    GammaName = '_gamma-8'

    NewtonIter = 270
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

    Gamma = 1e-10
    GammaName = '_gamma-10'

    NewtonIter = 270
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

if __name__ == '__main__':

    NewtonIter = 30
    Gamma = 1e-2
    GammaName = '_gamma-2'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

    NewtonIter = 90
    Gamma = 1e-3
    GammaName = '_gamma-3'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

    NewtonIter = 270
    Gamma = 1e-4
    GammaName = '_gamma-4'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

    NewtonIter = 810
    Gamma = 1e-5
    GammaName = '_gamma-5'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)



import asyncio
import json
import os
import websockets

AISSTREAM_API_KEY = "#YOUR_AISSTREAM_API_KEY"
SHIPS_FILE = "ships.json"
BOUNDING_BOX = [[-90, -180], [90, 180]]

def load_ships():
    if os.path.exists(SHIPS_FILE):
        with open(SHIPS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_ships(ships):
    with open(SHIPS_FILE, "w") as f:
        json.dump(ships, f, indent=4)

async def connect_ais_stream():
    ships = load_ships()
    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        subscribe_message = {
            "APIKey": AISSTREAM_API_KEY,
            "BoundingBoxes": [BOUNDING_BOX],
            "FilterMessageTypes": ["PositionReport"]
        }
        await websocket.send(json.dumps(subscribe_message))

        async for message_json in websocket:
            message = json.loads(message_json)
            message_type = message.get("MessageType")

            if message_type == "PositionReport":
                ais_message = message["Message"]["PositionReport"]
                metadata = message["MetaData"]

                mmsi = str(ais_message["UserID"])
                ship = {
                    "MMSI": mmsi,
                    "ShipName": metadata.get("ShipName", "Unknown").strip(),
                    "Latitude": ais_message["Latitude"],
                    "Longitude": ais_message["Longitude"],
                    "Speed": ais_message["Sog"],
                    "Course": ais_message["Cog"],
                    "Timestamp": metadata.get("time_utc")
                }

                ships[mmsi] = ship
                save_ships(ships)

                print("\n===== Live Ship Update =====")
                print("MMSI     :", ship["MMSI"])
                print("Name     :", ship["ShipName"])
                print("Latitude :", ship["Latitude"])
                print("Longitude:", ship["Longitude"])
                print("Speed    :", ship["Speed"], "knots")
                print("Course   :", ship["Course"])
                print("Time     :", ship["Timestamp"])

def track_live_ships():
    print("\nConnecting to live AIS stream... Press Ctrl+C to stop.")
    try:
        asyncio.run(connect_ais_stream())
    except KeyboardInterrupt:
        print("\nStopped live tracking.")

def view_tracked_ships():
    ships = load_ships()
    print("\n===== Tracked Ships =====")
    if not ships:
        print("No ships tracked yet.")
        return
    for mmsi, ship in ships.items():
        print("-" * 40)
        print("MMSI     :", ship["MMSI"])
        print("Name     :", ship["ShipName"])
        print("Latitude :", ship["Latitude"])
        print("Longitude:", ship["Longitude"])
        print("Speed    :", ship["Speed"], "knots")
        print("Course   :", ship["Course"])
        print("Last Seen:", ship["Timestamp"])

def search_ship_by_mmsi():
    ships = load_ships()
    mmsi = input("Enter MMSI: ")
    ship = ships.get(mmsi)
    if ship:
        print("\nShip Found")
        print("Name     :", ship["ShipName"])
        print("Latitude :", ship["Latitude"])
        print("Longitude:", ship["Longitude"])
        print("Speed    :", ship["Speed"], "knots")
        print("Course   :", ship["Course"])
        print("Last Seen:", ship["Timestamp"])
    else:
        print("Ship Not Found.")

def ship_tracking_menu():
    while True:
        print("\n===== Ship Tracking System =====")
        print("1. Track Live Ships (AIS Stream)")
        print("2. View Tracked Ships")
        print("3. Search Ship by MMSI")
        print("4. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":
            track_live_ships()
        elif choice == "2":
            view_tracked_ships()
        elif choice == "3":
            search_ship_by_mmsi()
        elif choice == "4":
            print("Exiting Ship Tracking System.")
            break
        else:
            print("Invalid Choice!")

if __name__ == "__main__":
    ship_tracking_menu()


import json
import os
import time
import difflib
import torch
import torch.nn as nn
import requests

MODEL_FILE = "dyslexia_model.pt"
RESULTS_FILE = "dyslexia_results.json"
ASSEMBLYAI_API_KEY = "#YOUR_ASSEMBLYAI_API_KEY"

WORD_LISTS = {
    "english": ["because", "friend", "necessary", "beautiful", "separate", "definitely", "receive", "believe"],
    "hindi": ["विद्यालय", "स्वतंत्रता", "आवश्यकता", "प्रतिनिधि", "संस्कृति"],
    "kannada": ["ಪ್ರಪಂಚ", "ಸ್ವಾತಂತ್ರ್ಯ", "ಆವಶ್ಯಕತೆ", "ಸಂಸ್ಕೃತಿ", "ಪ್ರತಿನಿಧಿ"]
}

REVERSAL_PAIRS = [("b", "d"), ("p", "q"), ("m", "w"), ("n", "u")]

class DyslexiaRiskNet(nn.Module):
    def __init__(self, input_size=6):
        super(DyslexiaRiskNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    model = DyslexiaRiskNet()
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    return model

def save_model(model):
    torch.save(model.state_dict(), MODEL_FILE)

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

def edit_distance_ratio(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def count_letter_reversals(word):
    count = 0
    for a, b in REVERSAL_PAIRS:
        count += word.count(a)
        count += word.count(b)
    return count

def select_language():
    print("\nSelect Language:")
    print("1. English")
    print("2. Hindi")
    print("3. Kannada")
    choice = input("Enter Choice: ")
    mapping = {"1": "english", "2": "hindi", "3": "kannada"}
    return mapping.get(choice, "english")

def predict_risk(features):
    model = load_model()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        risk = model(x).item()
    return risk

def spelling_test(user):
    language = select_language()
    words = WORD_LISTS[language]

    correct_count = 0
    total_similarity = 0
    total_reversals = 0
    total_time = 0

    print("\n===== Spelling Test =====")

    for word in words:
        print("\nListen carefully and type the word:", word)
        start_time = time.time()
        typed = input("Your Answer: ")
        elapsed = time.time() - start_time

        similarity = edit_distance_ratio(typed, word)
        reversals = count_letter_reversals(typed)

        total_similarity += similarity
        total_reversals += reversals
        total_time += elapsed

        if typed.strip().lower() == word.lower():
            correct_count += 1

    avg_similarity = total_similarity / len(words)
    avg_time = total_time / len(words)
    accuracy = correct_count / len(words)

    features = [accuracy, avg_similarity, total_reversals, avg_time, len(words), 1.0 if language != "english" else 0.0]
    risk_score = predict_risk(features)

    result = {
        "User": user,
        "Language": language,
        "Type": "Spelling",
        "Accuracy": accuracy,
        "AvgSimilarity": avg_similarity,
        "Reversals": total_reversals,
        "AvgTimePerWord": avg_time,
        "RiskScore": risk_score,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    results = load_results()
    results.append(result)
    save_results(results)

    print("\n===== Test Complete =====")
    print("Accuracy       :", round(accuracy * 100, 2), "%")
    print("Similarity     :", round(avg_similarity * 100, 2), "%")
    print("Reversal Errors:", total_reversals)
    print("Avg Time/Word  :", round(avg_time, 2), "sec")
    print("Dyslexia Risk  :", round(risk_score * 100, 2), "%")

def transcribe_speech(audio_file_path):
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    with open(audio_file_path, "rb") as f:
        upload_response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)

    audio_url = upload_response.json()["upload_url"]

    transcript_request = {"audio_url": audio_url}
    transcript_response = requests.post("https://api.assemblyai.com/v2/transcript", json=transcript_request, headers=headers)
    transcript_id = transcript_response.json()["id"]

    polling_endpoint = "https://api.assemblyai.com/v2/transcript/" + transcript_id

    while True:
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()["status"]

        if status == "completed":
            return polling_response.json()["text"]
        elif status == "error":
            return None

        time.sleep(3)

def reading_test(user):
    language = select_language()
    words = WORD_LISTS[language]
    passage = " ".join(words)

    print("\n===== Reading Test =====")
    print("Please read the following passage aloud and record it:")
    print(passage)

    audio_file_path = input("Enter path to recorded audio file: ")

    start_time = time.time()
    transcribed_text = transcribe_speech(audio_file_path)
    elapsed = time.time() - start_time

    if transcribed_text is None:
        print("Transcription failed.")
        return

    similarity = edit_distance_ratio(transcribed_text, passage)

    features = [similarity, similarity, 0, elapsed, len(words), 1.0 if language != "english" else 0.0]
    risk_score = predict_risk(features)

    result = {
        "User": user,
        "Language": language,
        "Type": "Reading",
        "Transcribed": transcribed_text,
        "Similarity": similarity,
        "TimeTaken": elapsed,
        "RiskScore": risk_score,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    results = load_results()
    results.append(result)
    save_results(results)

    print("\n===== Reading Test Complete =====")
    print("Transcribed Text:", transcribed_text)
    print("Similarity      :", round(similarity * 100, 2), "%")
    print("Time Taken      :", round(elapsed, 2), "sec")
    print("Dyslexia Risk   :", round(risk_score * 100, 2), "%")

def view_report(user):
    results = load_results()
    user_results = [r for r in results if r["User"] == user]

    print("\n===== Dyslexia Screening Report =====")

    if not user_results:
        print("No Test Records Found.")
        return

    for r in user_results:
        print("-" * 40)
        print("Language :", r["Language"])
        print("Type     :", r.get("Type", "Spelling"))
        print("Risk     :", round(r["RiskScore"] * 100, 2), "%")
        print("Timestamp:", r["Timestamp"])

def dyslexia_login():
    print("\n===== Dyslexia Screening Login =====")
    username = input("Enter Username: ")
    return username

def dyslexia_menu():
    user = dyslexia_login()

    while True:
        print("\n===== Dyslexia Screening System =====")
        print("1. Take Spelling Test")
        print("2. Take Reading Test")
        print("3. View My Report")
        print("4. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":
            spelling_test(user)
        elif choice == "2":
            reading_test(user)
        elif choice == "3":
            view_report(user)
        elif choice == "4":
            print("Thank you for using the Dyslexia Screening System!")
            break
        else:
            print("Invalid Choice!")

if __name__ == "__main__":
    dyslexia_menu()