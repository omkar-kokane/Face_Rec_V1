import cv2
import face_recognition
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import pickle
import os
from PIL import Image, ImageTk
import threading

class FaceRegistration:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Registration System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Variables
        self.current_frame = None
        self.is_capturing = False
        self.captured_faces = []
        self.person_name = ""
        
        # Create GUI elements
        self.setup_gui()
        
        # Start video feed
        self.update_frame()
        
    def setup_gui(self):
        """Setup the GUI elements"""
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="Face Registration System", 
                              font=("Arial", 20, "bold"), fg="blue")
        title_label.pack(pady=10)
        
        # Video frame container
        video_frame = tk.Frame(main_frame, bg="black", width=640, height=480)
        video_frame.pack(pady=10)
        video_frame.pack_propagate(False)
        
        # Video label
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(expand=True)
        
        # Status frame
        status_frame = tk.Frame(main_frame)
        status_frame.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(status_frame, text="Click 'Start Registration' to begin", 
                                    font=("Arial", 12), fg="blue")
        self.status_label.pack()
        
        # Progress label
        self.progress_label = tk.Label(status_frame, text="Photos captured: 0/5", 
                                      font=("Arial", 10), fg="gray")
        self.progress_label.pack()
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Create buttons in a grid layout
        self.start_btn = tk.Button(button_frame, text="Start Registration", 
                                  command=self.start_registration,
                                  font=("Arial", 12, "bold"), bg="green", fg="white",
                                  padx=20, pady=10, width=15)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.capture_btn = tk.Button(button_frame, text="Capture Photo", 
                                    command=self.capture_photo,
                                    font=("Arial", 12, "bold"), bg="blue", fg="white",
                                    padx=20, pady=10, width=15, state=tk.DISABLED)
        self.capture_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.finish_btn = tk.Button(button_frame, text="Finish Registration", 
                                   command=self.finish_registration,
                                   font=("Arial", 12, "bold"), bg="orange", fg="white",
                                   padx=20, pady=10, width=15, state=tk.DISABLED)
        self.finish_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.exit_btn = tk.Button(button_frame, text="Exit", 
                                 command=self.exit_application,
                                 font=("Arial", 12, "bold"), bg="red", fg="white",
                                 padx=20, pady=10, width=15)
        self.exit_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Instructions
        instructions = tk.Label(main_frame, 
                               text="Instructions:\n1. Click 'Start Registration' and enter your name\n2. Position your face in the camera\n3. Click 'Capture Photo' 5 times from different angles\n4. Click 'Finish Registration' when done",
                               font=("Arial", 10), justify=tk.LEFT, fg="darkgreen")
        instructions.pack(pady=10)
        
    def update_frame(self):
        """Update the video frame continuously"""
        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Detect faces and draw rectangles
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Draw rectangles around faces
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert frame to display in tkinter
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            
            # Update face detection status
            if len(face_locations) > 0:
                self.status_label.config(text="Face detected! Ready to capture", fg="green")
            else:
                self.status_label.config(text="No face detected. Position your face in the camera", fg="red")
        
        # Schedule next frame update
        self.root.after(10, self.update_frame)
    
    def start_registration(self):
        """Start the registration process"""
        # Get person name
        self.person_name = simpledialog.askstring("Person Name", "Enter the person's name:")
        
        if not self.person_name:
            messagebox.showwarning("Warning", "Name is required for registration!")
            return
        
        # Check if name already exists
        if self.check_name_exists(self.person_name):
            result = messagebox.askyesno("Name Exists", 
                                       f"'{self.person_name}' already exists. Do you want to update their photos?")
            if not result:
                return
        
        # Reset variables
        self.captured_faces = []
        self.is_capturing = True
        
        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.NORMAL)
        self.finish_btn.config(state=tk.DISABLED)
        
        self.status_label.config(text=f"Registration started for: {self.person_name}", fg="blue")
        self.progress_label.config(text="Photos captured: 0/5")
        
        messagebox.showinfo("Registration Started", 
                          f"Registration started for {self.person_name}.\n"
                          "Please capture 5 photos from different angles.\n"
                          "Make sure your face is clearly visible in each photo.")
    
    def capture_photo(self):
        """Capture a photo for registration"""
        if not self.is_capturing:
            return
        
        if self.current_frame is None:
            messagebox.showerror("Error", "No camera frame available!")
            return
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            messagebox.showwarning("Warning", "No face detected! Please position your face in the camera.")
            return
        
        if len(face_locations) > 1:
            messagebox.showwarning("Warning", "Multiple faces detected! Please ensure only one person is in the frame.")
            return
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) > 0:
            self.captured_faces.append(face_encodings[0])
            
            # Update progress
            progress = len(self.captured_faces)
            self.progress_label.config(text=f"Photos captured: {progress}/5")
            
            # Show success message
            self.status_label.config(text=f"Photo {progress} captured successfully!", fg="green")
            
            # Enable finish button if we have enough photos
            if progress >= 3:  # Minimum 3 photos required
                self.finish_btn.config(state=tk.NORMAL)
            
            # Auto-finish if we have 5 photos
            if progress >= 5:
                self.status_label.config(text="All photos captured! You can finish registration now.", fg="blue")
                messagebox.showinfo("Complete", "5 photos captured! You can now finish the registration.")
        else:
            messagebox.showerror("Error", "Could not encode face. Please try again.")
    
    def finish_registration(self):
        """Finish the registration process and save data"""
        if len(self.captured_faces) < 3:
            messagebox.showwarning("Warning", "Please capture at least 3 photos before finishing registration.")
            return
        
        try:
            # Load existing data
            face_data = self.load_face_data()
            
            # Add or update person data
            face_data[self.person_name] = self.captured_faces
            
            # Save updated data
            self.save_face_data(face_data)
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"Registration completed for {self.person_name}!\n"
                              f"Captured {len(self.captured_faces)} photos.")
            
            # Reset UI
            self.reset_registration()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save registration data: {str(e)}")
    
    def load_face_data(self):
        """Load existing face data from file"""
        data_file = "data/face_encodings.pkl"
        if os.path.exists(data_file):
            try:
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_face_data(self, face_data):
        """Save face data to file"""
        data_file = "data/face_encodings.pkl"
        os.makedirs("data", exist_ok=True)
        with open(data_file, 'wb') as f:
            pickle.dump(face_data, f)
    
    def check_name_exists(self, name):
        """Check if a name already exists in the database"""
        face_data = self.load_face_data()
        return name in face_data
    
    def reset_registration(self):
        """Reset the registration UI"""
        self.is_capturing = False
        self.captured_faces = []
        self.person_name = ""
        
        # Reset UI elements
        self.start_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.finish_btn.config(state=tk.DISABLED)
        
        self.status_label.config(text="Registration completed! Ready for next person.", fg="green")
        self.progress_label.config(text="Photos captured: 0/5")
    
    def exit_application(self):
        """Exit the application"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceRegistration()
    app.run()