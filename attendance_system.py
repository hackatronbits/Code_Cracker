import cv2
import dlib
import numpy as np
import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import geocoder
import firebase_admin
from firebase_admin import credentials, firestore





cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# Initialize face detector and recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model.dat')

class AttendanceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Attendance System")
        self.stop_flag = False
        self.session_logged_names = set()  # Track logged names for the current session
        
        # Allowed coordinates (latitude, longitude) for attendance; admin can update here
        self.allowed_coordinates = [(23.6546, 86.4737)]  # Default example coordinate
        self.allowed_radius_meters = 1000000
        
        # UI Buttons for role selection
        self.admin_button = tk.Button(master, text="Admin Login", command=self.admin_login)
        self.admin_button.pack(pady=10)

        self.student_button = tk.Button(master, text="Student Login", command=self.student_login)
        self.student_button.pack(pady=10)

    def admin_login(self):
        admin_window = tk.Toplevel(self.master)
        admin_window.title("Admin Panel")
        admin_window.geometry('350x200')

        tk.Label(admin_window, text="Enter Allowed Coordinate (lat, lon):").pack(pady=5)
        self.coord_entry = tk.Entry(admin_window, width=30)
        self.coord_entry.pack(pady=5)
        self.coord_entry.insert(0, f"{self.allowed_coordinates[0][0]}, {self.allowed_coordinates[0][1]}")

        tk.Button(admin_window, text="Set Coordinates", command=self.set_coordinates).pack(pady=5)
        tk.Button(admin_window, text="Set Attendance Range", command=self.set_attendance_range).pack(pady=5)
        tk.Button(admin_window, text="View Attendance Log", command=self.view_log).pack(pady=5)
        tk.Button(admin_window, text="Close", command=admin_window.destroy).pack(pady=5)

    def set_coordinates(self):
        coords = self.coord_entry.get()
        try:
            lat, lon = map(float, coords.split(","))
            self.allowed_coordinates = [(lat, lon)]
            messagebox.showinfo("Success", "Coordinates set successfully.")
        except ValueError:
            messagebox.showerror("Error", "Invalid coordinates format. Enter as: lat, lon")

    def set_attendance_range(self):
        range_window = tk.Toplevel(self.master)
        range_window.title("Set Attendance Range")
        range_window.geometry('300x150')
        tk.Label(range_window, text="Enter Radius in Meters:").pack(pady=5)
        self.radius_entry = tk.Entry(range_window, width=10)
        self.radius_entry.pack(pady=5)
        self.radius_entry.insert(0, str(self.allowed_radius_meters))
        tk.Button(range_window, text="Set Radius", command=self.update_radius).pack(pady=5)
        tk.Button(range_window, text="Close", command=range_window.destroy).pack(pady=5)

    def update_radius(self):
        try:
            radius = float(self.radius_entry.get())
            if radius > 0:
                self.allowed_radius_meters = radius
                messagebox.showinfo("Success", "Attendance range updated successfully.")
            else:
                messagebox.showerror("Error", "Radius must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Invalid radius input.")
            

    def student_login(self):
        student_window = tk.Toplevel(self.master)
        student_window.title("Student Panel")
        student_window.geometry('300x200')

        tk.Button(student_window, text="Start Attendance", command=self.start_attendance).pack(pady=10)
        tk.Button(student_window, text="Speech Roll Call", command=self.speech_roll_call).pack(pady=10)
        tk.Button(student_window, text="View Attendance Log", command=self.view_log).pack(pady=10)
        tk.Button(student_window, text="Close", command=student_window.destroy).pack(pady=10)

    def train_model(self, data_folder='attendance_data'):
        known_face_encodings = []
        known_face_names = []
        if not os.path.exists(data_folder):
            messagebox.showerror("Error", f"Data folder '{data_folder}' does not exist.")
            return known_face_encodings, known_face_names
        
        for student_folder in os.listdir(data_folder):
            student_path = os.path.join(data_folder, student_folder)
            if not os.path.isdir(student_path):
                continue
            for image_name in os.listdir(student_path):
                image_path = os.path.join(student_path, image_name)
                try:
                    img = dlib.load_rgb_image(image_path)
                except Exception:
                    continue  # skip corrupt images
                dets = detector(img, 1)
                for d in dets:
                    shape = sp(img, d)
                    face_encoding = facerec.compute_face_descriptor(img, shape)
                    known_face_encodings.append(np.array(face_encoding))
                    known_face_names.append(student_folder)
        return known_face_encodings, known_face_names

    def log_attendance(self, face_names, location):
        if not os.path.exists('attendance_log.csv'):
            with open('attendance_log.csv', 'w') as f:
                f.write("Name,Timestamp,Location\n")
        with open('attendance_log.csv', 'a') as f:
            for name in face_names:
                if name != "Unknown" and name not in self.session_logged_names:
                    f.write(f"{name},{datetime.now()},{location}\n")
                    self.session_logged_names.add(name)

    def get_current_location(self):
        try:
            g = geocoder.ip('me')
            if g.ok and g.latlng:
                return tuple(g.latlng)
            else:
                return None
        except Exception:
            return None

    def haversine_distance(self, coord1, coord2):
        import math
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (math.sin(delta_phi/2) ** 2 +
             math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2)
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def is_within_allowed_coordinates(self, location):
        for coord in self.allowed_coordinates:
            if self.haversine_distance(coord, location) <= self.allowed_radius_meters:
                return True
        return False

    def capture_attendance(self, known_face_encodings, known_face_names):
        self.stop_flag = False
        self.session_logged_names.clear()
        video_capture = cv2.VideoCapture(0)

        while not self.stop_flag:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = detector(rgb_frame, 1)
            face_encodings = [np.array(facerec.compute_face_descriptor(rgb_frame, sp(rgb_frame, d))) for d in face_locations]

            face_names = []
            for face_encoding in face_encodings:
                if len(known_face_encodings) == 0:
                    name = "Unknown"
                else:
                    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                    min_distance = min(distances) if len(distances) > 0 else float('inf')
                    if min_distance < 0.6:
                        best_match_index = np.argmin(distances)
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                face_names.append(name)

            location = self.get_current_location()
            location_str = f"{location[0]:.6f},{location[1]:.6f}" if location else "Unknown"

            if location and self.is_within_allowed_coordinates(location):
                self.log_attendance(face_names, location_str)
            else:
                messagebox.showwarning("Location Warning", "You are outside the allowed location area. Attendance not logged.")
                self.stop_flag = True
                break

            for (d, name) in zip(face_locations, face_names):
                left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Attendance System - Press q to stop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_flag = True
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def start_attendance(self):
        res = messagebox.askyesno("Enable Location", "Enable location to verify attendance area? (Required)")
        if not res:
            messagebox.showinfo("Location Disabled", "Attendance cannot be started without location enabled.")
            return

        location = self.get_current_location()
        if not location:
            messagebox.showerror("Location Error", "Could not get current location. Check internet or location settings.")
            return

        if not self.is_within_allowed_coordinates(location):
            messagebox.showwarning("Location Error", "Current location outside allowed area. Attendance cannot start.")
            return

        known_face_encodings, known_face_names = self.train_model()
        if len(known_face_encodings) == 0:
            messagebox.showerror("Error", "No training face data found. Please add data in 'attendance_data' folder.")
            return

        threading.Thread(target=self.capture_attendance, args=(known_face_encodings, known_face_names), daemon=True).start()
        messagebox.showinfo("Attendance Started", "Attendance capture started. Press 'q' in camera window to stop.")

    def stop_attendance(self):
        self.stop_flag = True
        messagebox.showinfo("Info", "Attendance capture stopped.")

    def view_log(self):
        if not os.path.exists('attendance_log.csv'):
            messagebox.showinfo("Attendance Log", "No attendance records found.")
            return
        with open('attendance_log.csv', 'r') as f:
            log_data = f.read()
        log_window = tk.Toplevel(self.master)
        log_window.title("Attendance Log")
        log_text = tk.Text(log_window, width=70, height=25)
        log_text.pack()
        log_text.insert(tk.END, log_data)
        log_text.config(state=tk.DISABLED)

    def speech_roll_call(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            messagebox.showinfo("Speech Recognition", "Listening...")
            audio = recognizer.listen(source)
        try:
            name = recognizer.recognize_google(audio)
            if name not in self.session_logged_names:
                location = self.get_current_location()
                location_str = f"{location[0]:.6f},{location[1]:.6f}" if location else "Unknown"
                if location and self.is_within_allowed_coordinates(location):
                    self.log_attendance([name], location_str)
                    messagebox.showinfo("Speech Recognition", f"Attendance marked for {name} at location {location_str}")
                else:
                    messagebox.showwarning("Location Warning", "You are outside the allowed location area. Attendance not logged.")
            else:
                messagebox.showinfo("Speech Recognition", f"{name} is already marked present.")
        except sr.UnknownValueError:
            messagebox.showerror("Speech Recognition", "Could not understand audio.")
        except sr.RequestError:
            messagebox.showerror("Speech Recognition", "Could not request results.")
            

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
