import xml.etree.ElementTree as ET

def correctTag(filename, parentTag, childTag, value="Unspecified"):
    tree = ET.parse(filename)
    P = tree.findall(parentTag)
    for p in P:
        if p.find(childTag) == None:
            c = ET.SubElement(p, childTag)
            c.text = value
        name = p.find("name")
        if name.text == "just_a_person":
            name.text = "without_mask"
        bndbox = p.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        bndbox.find("xmin").text = f"{round(xmin)}"
        bndbox.find("ymin").text = f"{round(ymin)}"
        bndbox.find("xmax").text = f"{round(xmax)}"
        bndbox.find("ymax").text = f"{round(ymax)}"
    tree.write(filename)

def corrects(filenames, parentTag, childTag, value="Unspecified"):
    for filename in filenames:
        correctTag(filename, parentTag, childTag, value)