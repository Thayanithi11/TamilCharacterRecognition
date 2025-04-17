import React from 'react';

export default function Convert(props) {
  const {processedImg,selectedFile,handleFileChange,handleUpload,handleEnhance,enhancedImg}=props
  return (
<div className='convert-page'>
<div className="btn-cntnr">
  <div className="input">
    <input type="file" accept="image/*" onChange={handleFileChange} />
  </div>
  <button className="btn" onClick={handleUpload}>Submit</button>
</div>

{selectedFile && (
  <div className="img-cntnr">
    <img className="contained" src={URL.createObjectURL(selectedFile)} alt="Original" />
  </div>
)}

{processedImg && (
  <>
    <div className="pimg-cntnr">
      <img src={processedImg} className="contained" alt="Processed image" />
    </div>
    <div className="btn-cntnr">
      <button className="btn" onClick={handleEnhance}>Enhance</button>
    </div>
  </>
)}

{enhancedImg && (
  <div className="pimg-cntnr">
    <img src={enhancedImg} className="contained" alt="Enhanced image" />
  </div>
)}
</div>
);
}
