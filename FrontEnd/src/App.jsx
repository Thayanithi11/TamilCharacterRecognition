import { useState } from 'react'
import Header from "./Components/Header"
import Main from "./Components/Main"
import Convert from "./Components/Convert"

function App() {
     const [convert,useConvert]=useState(0)
     const [selectedFile,setSelectedFile]=useState(null)
     const [processedImg,setProcessedImg]=useState(null)
     const [enhancedImg, setEnhancedImg] = useState(null);
  
     const handleFileChange = (event) => {
      setSelectedFile(event.target.files[0]);
      console.log("Event done")
    };

    const handleUpload = async () => {
      if (!selectedFile) {
          alert("Please select a file first!");
          return;
      }

      const formData = new FormData();
      formData.append("image", selectedFile);

      try {
          const response = await fetch("http://localhost:5000/upload", {
              method: "POST",
              body: formData,
          });

          if (!response.ok) throw new Error("Failed to process image");

          const imageBlob = await response.blob();
          const imageUrl = URL.createObjectURL(imageBlob);
          setProcessedImg(imageUrl);
      } catch (error) {
          console.error("Error:", error);
      }
  };

  const handleEnhance = async () => {
    try {
        const response = await fetch("http://localhost:5000/enhance", {
            method: "GET",
        });

        if (!response.ok) throw new Error("Failed to enhance image");

        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);
        setEnhancedImg(imageUrl);
    } catch (error) {
        console.error("Enhancement Error:", error);
    }
};

 return (
    <>
      <Header handleConvert={useConvert}/>
      {convert===0?<Main/>:<Convert processedImg={processedImg} handleFileChange={handleFileChange}
       selectedFile={selectedFile} handleUpload={handleUpload} handleEnhance={handleEnhance} enhancedImg={enhancedImg}/>} 
    </>
  )
}

export default App
