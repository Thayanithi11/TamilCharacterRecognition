import React from 'react'

export default function Header(props) {
      const {handleConvert}=props

  return (
    <div>
       <div className='header'>
          <div className="l-header">
          <img src="../src/assets/sastra_img.png" className='sastraimg'/>
          <img src='../src/assets/TnGovernment.png' className='tngimg'/>
      </div>
      <div className="m-header">
         <h6 className='title'>Tamil Palm Text converter</h6>
      </div>
      <div className="r-header">
          <a href="/" onClick={()=>{
              handleConvert(0)
          }} 
          className='mrgn'>Home</a>
          <a className='mrgn convert' onClick={()=>{
             handleConvert(1)
          }}>Convert</a> 
           <p className='mrgn'>Scholars</p>
           <p className='mrgn'>Heritage</p>
      </div>
    </div>
    </div>
  )
}
