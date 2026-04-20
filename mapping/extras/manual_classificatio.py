#Manually classify clips

import cv2
import os
import shutil
import json

json_folder = r"C:\\Users\\HV\Desktop\\bruno_work\\save_electrospray\\dataset\\current\\EW82\\unclassified\\"
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]


with open(f"{json_folder}/{json_files[1]}", 'r') as file:
    data = json.load(file)
    print(f"File opened: {file}")

input_folder = r"C:\\Users\\HV\Desktop\\bruno_work\\save_electrospray\\dataset\\images\\EW82\\unclassified"
output_base = r"C:\\Users\\HV\Desktop\\bruno_work\\save_electrospray\\dataset\\images\\EW82"


classes = ["cone_jet", "dripping", "intermitent", "multi_jet", "delete", "undefined"]

for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

videos = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
videos.sort()

print("Controls:")
print("1–6 → assign class")
print("q   → quit")
print("n   → skip video")

sample = -1
for video_name in videos:
    print(f"sample {sample}")
    video_path = os.path.join(input_folder, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening:", video_name)
        continue

    print(f"\nLabeling: {video_name}")
    sample += 1
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow("Video", frame)

        key = cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF

        if key == ord('q'):
            with open(f"{json_folder}/{json_files[0]}", 'w') as file:
                # 'indent=4' makes the JSON file easy for humans to read
                json.dump(data, file, indent=4)
            cap.release()
            cv2.destroyAllWindows()
            exit()

        elif key == ord('n'):
            print("Skipped")
            break

        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            class_index = int(chr(key)) - 1
            destination = os.path.join(output_base, classes[class_index], video_name)
            print(f"Sample {sample}: ")
            print("From "+ data[f"sample {sample}"]["spray_mode"])
            data[f"sample {sample}"]["spray_mode"] = classes[class_index]
            print("To "+ data[f"sample {sample}"]["spray_mode"])
            cap.release()
            cv2.destroyAllWindows()
            shutil.move(video_path, destination)
            print(f"Moved to {classes[class_index]}")
            break
    cap.release()
    cv2.destroyAllWindows()


with open(f"{json_folder}/{json_files[0]}", 'w') as file:
    # 'indent=4' makes the JSON file easy for humans to read
    json.dump(data, file, indent=4)
    

print("Done labeling.")
