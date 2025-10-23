import json, os, argparse

GALLERY = "gallery.json"

def remove_embedding(person_id):
    if not os.path.exists(GALLERY):
        print("Gallery file not found.")
        return

    with open(GALLERY, "r") as f:
        data = json.load(f)

    if person_id not in data:
        print(f"{person_id} not found in gallery.")
        return

    del data[person_id]

    with open(GALLERY, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Unenrolled {person_id}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Employee/person ID to remove")
    args = ap.parse_args()
    remove_embedding(args.id)

if __name__ == "__main__":
    main()
