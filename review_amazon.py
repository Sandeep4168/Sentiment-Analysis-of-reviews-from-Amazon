# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:43:08 2020

@author: user
"""


import scrapy

from ..items import ReviewAmazonItem


class ReviewAmzonSpider(scrapy.Spider):
    name='review'
    pno=3
    urls=["https://www.amazon.in/Test-Exclusive-748/product-reviews/B07DJLVJ5M/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_1_1805560031&pageNumber="
                ,"https://www.amazon.in/Redmi-Note-Neptune-Blue-Storage/product-reviews/B07X1KT6LD/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_2_1805560031&pageNumber="
                ,"https://www.amazon.in/Redmi-Black-3GB-32GB-Storage/product-reviews/B077Q7F7Z3/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_3_1805560031&pageNumber="
                ,"https://www.amazon.in/Test-Exclusive-738/product-reviews/B07DJL15MJ/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_4_1805560031&pageNumber="
                ,"https://www.amazon.in/Test-Exclusive-720/product-reviews/B07DJLVHYC/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_5_1805560031&pageNumber="
                ,"https://www.amazon.in/Redmi-Y3-Elegant-Storage-4000mAH/product-reviews/B07QS3VR4T/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_6_1805560031&pageNumber="
                ,"https://www.amazon.in/Fluorite-Purple-Storage-Additional-Exchange/product-reviews/B07PTLD8P9/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_7_1805560031&pageNumber="
                ,"https://www.amazon.in/Samsung-Galaxy-Electric-128GB-Storage/product-reviews/B085J1J32G/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_8_1805560031&pageNumber="
                ,"https://www.amazon.in/Samsung-Galaxy-Ocean-128GB-Storage/product-reviews/B07HGGYWL6/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_9_1805560031&pageNumber="
                ,"https://www.amazon.in/Test-Exclusive-614/product-reviews/B07HGJJ559/ref=cm_cr_arp_d_paging_btm_next_2?ref_=fspcr_pl_ar_10_1805560031&pageNumber="]
    start_urls=[]
    for url in urls:
        for i in range(100):
            start_urls.append(url+str(i))
        

           
           
    
    def parse(self,response):
        
        items=ReviewAmazonItem()
        name=response.css('.a-text-ellipsis .a-link-normal ::text').getall()
        review=response.css('.review-text-content ::text').getall()
        ratings=response.css('.review-rating ::text').getall()
        
        
        items['name']=name
        items['review']=review
        items['ratings']=ratings
        
        
        yield items
        
        
        
            
            
        
        
        
        
        
       
        
    
    